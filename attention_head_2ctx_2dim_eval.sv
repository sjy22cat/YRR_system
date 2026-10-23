`timescale 1ns/1ps

// Two-context by two-dimension-block single-head Attention evaluator.
// The input stream remains one [TILE_SIZE x HEAD_DIM] K/V tile per handshake.
// Two consecutive context tiles are buffered, evaluated in parallel, and
// merged through local (m,l,z) online-Softmax statistics.  Within each tile,
// two head-dimension blocks are evaluated in parallel.  A partial QK block is
// therefore never sent to Softmax.
module attention_head_2ctx_2dim_eval #(
    parameter integer HEAD_DIM          = 128,
    parameter integer TILE_SIZE         = 32,
    parameter integer MAX_CONTEXT       = 1152,
    parameter [31:0] SCALE_BITS         = 32'h3db504f3,
    parameter integer PRE_SCALE_QUERY   = 1,
    parameter integer SKIP_IDENTITY_RESCALE = 1,
    parameter integer PIPELINE_SOFTMAX  = 1,
    parameter integer PIPELINE_MAC      = 1,
    parameter integer SOFTMAX_SERIAL    = 0,
    parameter integer BLOCK_PARALLEL    = 2,
    parameter integer CONTEXT_PARALLEL  = 2
) (
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             i_cmd_valid,
    output wire                             o_cmd_ready,
    input  wire [10:0]                      i_context_len,
    input  wire [HEAD_DIM*32-1:0]           i_query,
    input  wire                             i_kv_valid,
    output wire                             o_kv_ready,
    input  wire [TILE_SIZE*HEAD_DIM*32-1:0] i_key_tile,
    input  wire [TILE_SIZE*HEAD_DIM*32-1:0] i_value_tile,
    output wire                             o_valid,
    input  wire                             i_ready,
    output wire [HEAD_DIM*32-1:0]           o_result,
    output reg                              o_error,
    output reg [31:0]                       o_cycle_count
);
    localparam integer QK_CHUNKS       = HEAD_DIM / TILE_SIZE;
    localparam integer OUT_CHUNKS      = HEAD_DIM / TILE_SIZE;
    localparam integer QK_GROUPS       = QK_CHUNKS / BLOCK_PARALLEL;
    localparam integer OUT_GROUPS      = OUT_CHUNKS / BLOCK_PARALLEL;
    localparam integer TREE_LEVELS     = $clog2(TILE_SIZE);
    localparam integer MAX_TREE_LEVELS = 5;
    localparam integer MAX_GROUP_LEVELS = (TREE_LEVELS < 2) ? TREE_LEVELS : 2;
    localparam integer MAX_GROUPS      = TILE_SIZE >> MAX_GROUP_LEVELS;
    localparam integer MAX_REMAIN_GROUPS = (MAX_GROUPS > 2) ? (MAX_GROUPS/2) : 1;
    localparam integer MAX_SUM_REMAIN_GROUPS = (MAX_GROUPS > 1) ? (MAX_GROUPS/2) : 1;
    localparam integer MAX_REMAIN_LEVELS =
        (MAX_REMAIN_GROUPS <= 1) ? 0 : $clog2(MAX_REMAIN_GROUPS);
    localparam integer MAX_SUM_REMAIN_LEVELS =
        (MAX_SUM_REMAIN_GROUPS <= 1) ? 0 : $clog2(MAX_SUM_REMAIN_GROUPS);
    localparam [31:0] FP_ONE           = 32'h3f800000;
    localparam [31:0] FP_ZERO          = 32'h00000000;
    localparam [31:0] FP_NEG_INF       = 32'hff800000;

    localparam integer S_IDLE          = 0;
    localparam integer S_QUERY_SCALE   = 1;
    localparam integer S_WAIT_TILE     = 2;
    localparam integer S_QK_MUL        = 3;
    localparam integer S_QK_R1         = 4;
    localparam integer S_QK_R2         = 5;
    localparam integer S_QK_R3         = 6;
    localparam integer S_QK_R4         = 7;
    localparam integer S_QK_R5         = 8;
    localparam integer S_QK_GROUP      = 9;
    localparam integer S_QK_ACC        = 10;
    localparam integer S_SM_MAX        = 11;
    localparam integer S_SM_MAX_R1     = 12;
    localparam integer S_SM_MAX_R2     = 13;
    localparam integer S_SM_EXP        = 17;
    localparam integer S_SM_SUM        = 18;
    localparam integer S_SM_SUM_R1     = 19;
    localparam integer S_SM_SUM_R2     = 20;
    localparam integer S_SM_SUM_R3     = 21;
    localparam integer S_SM_SUM_R4     = 22;
    localparam integer S_SM_SUM_R5     = 23;
    localparam integer S_PV_MUL        = 24;
    localparam integer S_PV_R1         = 25;
    localparam integer S_PV_R2         = 26;
    localparam integer S_PV_R3         = 27;
    localparam integer S_PV_R4         = 28;
    localparam integer S_PV_R5         = 29;
    localparam integer S_PV_ACC        = 30;
    localparam integer S_PAIR_MAX      = 31;
    localparam integer S_PAIR_EXP      = 32;
    localparam integer S_PAIR_SCALE    = 33;
    localparam integer S_PAIR_SUM      = 34;
    localparam integer S_GLOBAL_SCALE  = 35;
    localparam integer S_GLOBAL_UPDATE  = 36;
    localparam integer S_FINAL_INV     = 37;
    localparam integer S_FINAL_SCALE   = 38;
    localparam integer S_RESPONSE      = 39;
    localparam integer S_SM_DIFF       = 40;
    localparam integer S_PAIR_DIFF     = 41;

    reg [5:0] state;
    reg [10:0] context_len_q, processed_keys_q;
    reg [10:0] qk_group_q, pv_group_q;
    reg load_lane_q;
    reg [CONTEXT_PARALLEL-1:0] tile_valid_q;
    reg [TILE_SIZE-1:0] token_valid_q [0:CONTEXT_PARALLEL-1];
    reg [HEAD_DIM*32-1:0] query_q, query_scaled_q;
    reg [TILE_SIZE*HEAD_DIM*32-1:0] key_q [0:CONTEXT_PARALLEL-1];
    reg [TILE_SIZE*HEAD_DIM*32-1:0] value_q [0:CONTEXT_PARALLEL-1];
    reg [HEAD_DIM*32-1:0] result_q;

    reg [31:0] score_accum [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    reg [31:0] score_final [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    reg [31:0] beta_q [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    reg [31:0] beta_diff_q [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    reg [31:0] pv_root_q [0:CONTEXT_PARALLEL-1][0:BLOCK_PARALLEL-1][0:TILE_SIZE-1];
    reg [31:0] local_max_q [0:CONTEXT_PARALLEL-1];
    reg [31:0] local_sum_q [0:CONTEXT_PARALLEL-1];
    reg [31:0] local_z_q [0:CONTEXT_PARALLEL-1][0:HEAD_DIM-1];

    reg [31:0] running_max_q, running_sum_q;
    reg [31:0] output_accum [0:HEAD_DIM-1];
    reg have_context_q;
    reg [31:0] pair_max_q, new_max_q;
    reg [31:0] pair_alpha_diff_q [0:CONTEXT_PARALLEL-1];
    reg [31:0] global_alpha_diff_q, pair_beta_diff_q;
    reg [31:0] pair_alpha_q [0:CONTEXT_PARALLEL-1];
    reg [31:0] pair_beta_q, global_alpha_q;
    reg [31:0] pair_sum_prod_q [0:CONTEXT_PARALLEL-1];
    reg [31:0] pair_z_prod_q [0:CONTEXT_PARALLEL-1][0:HEAD_DIM-1];
    reg [31:0] pair_sum_q;
    reg [31:0] pair_z_q [0:HEAD_DIM-1];
    reg [31:0] global_sum_prod_q, pair_sum_global_prod_q;
    reg [31:0] inverse_sum_q;

    reg [31:0] mac_tree_q [0:CONTEXT_PARALLEL-1][0:BLOCK_PARALLEL-1]
        [0:MAX_TREE_LEVELS][0:TILE_SIZE-1][0:TILE_SIZE-1];
    reg [31:0] sm_max_tree_q [0:CONTEXT_PARALLEL-1]
        [0:MAX_TREE_LEVELS][0:TILE_SIZE-1];
    reg [31:0] sm_sum_tree_q [0:CONTEXT_PARALLEL-1]
        [0:MAX_TREE_LEVELS][0:TILE_SIZE-1];

    integer init_c, init_b, init_i, init_j, init_l;
    integer c, b, p, d, l;

    function automatic [31:0] fp_max(input [31:0] a, input [31:0] bval);
        begin
            if (a[31] != bval[31])
                fp_max = a[31] ? bval : a;
            else if (a[31])
                fp_max = (a[30:0] < bval[30:0]) ? a : bval;
            else
                fp_max = (a[30:0] > bval[30:0]) ? a : bval;
        end
    endfunction

    initial begin
        if (HEAD_DIM < 1 || TILE_SIZE < 2 || TILE_SIZE > 32 ||
            MAX_CONTEXT < 1 || MAX_CONTEXT > 2047 ||
            HEAD_DIM % TILE_SIZE != 0 || QK_CHUNKS % BLOCK_PARALLEL != 0 ||
            (TILE_SIZE & (TILE_SIZE-1)) != 0 ||
            BLOCK_PARALLEL != 2 || CONTEXT_PARALLEL != 2 ||
            PRE_SCALE_QUERY < 0 || PRE_SCALE_QUERY > 1 ||
            PIPELINE_SOFTMAX < 0 || PIPELINE_SOFTMAX > 1 ||
            PIPELINE_MAC < 0 || PIPELINE_MAC > 1 ||
            SOFTMAX_SERIAL < 0 || SOFTMAX_SERIAL > 1)
            $fatal(1, "Invalid 2x2 Attention tile configuration");
    end

    assign o_cmd_ready = (state == S_IDLE);
    assign o_kv_ready  = (state == S_WAIT_TILE);
    assign o_valid     = (state == S_RESPONSE);
    assign o_result    = result_q;

    wire token_valid [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    genvar tvc, tv;
    generate
        for (tvc = 0; tvc < CONTEXT_PARALLEL; tvc = tvc + 1) begin : g_token_ctx
            for (tv = 0; tv < TILE_SIZE; tv = tv + 1) begin : g_token
                // Capture context bounds with each accepted tile.  Later
                // datapath stages consume this registered mask, so the
                // context-length comparator is outside the MAC/Softmax path.
                assign token_valid[tvc][tv] = tile_valid_q[tvc] &&
                    token_valid_q[tvc][tv];
            end
        end
    endgenerate

    wire [31:0] query_scale_wire [0:HEAD_DIM-1];
    genvar qs;
    generate
        for (qs = 0; qs < HEAD_DIM; qs = qs + 1) begin : g_query_scale
            wire [7:0] query_scale_status;
            DW_fp_mult u_query_scale(
                .a(query_q[qs*32 +: 32]), .b(SCALE_BITS), .rnd(3'b000),
                .z(query_scale_wire[qs]), .status(query_scale_status));
        end
    endgenerate

    // Context tile x head-dimension block x 32-token/32-dimension MAC plane.
    wire [31:0] mac_product [0:CONTEXT_PARALLEL-1][0:BLOCK_PARALLEL-1]
        [0:TILE_SIZE-1][0:TILE_SIZE-1];
    genvar mc, mb, mp, ml;
    generate
        for (mc = 0; mc < CONTEXT_PARALLEL; mc = mc + 1) begin : g_mac_ctx
            for (mb = 0; mb < BLOCK_PARALLEL; mb = mb + 1) begin : g_mac_blk
                for (mp = 0; mp < TILE_SIZE; mp = mp + 1) begin : g_mac_row
                    for (ml = 0; ml < TILE_SIZE; ml = ml + 1) begin : g_mac_lane
                        wire [31:0] qk_key_word = key_q[mc][
                            (mp*HEAD_DIM +
                             (qk_group_q*BLOCK_PARALLEL+mb)*TILE_SIZE+ml)*32 +: 32];
                        wire [31:0] pv_value_word = value_q[mc][
                            (ml*HEAD_DIM +
                             (pv_group_q*BLOCK_PARALLEL+mb)*TILE_SIZE+mp)*32 +: 32];
                        wire [31:0] mac_a = (state == S_QK_MUL) ?
                            query_scaled_q[
                                ((qk_group_q*BLOCK_PARALLEL+mb)*TILE_SIZE+ml)*32 +: 32] :
                            (state == S_PV_MUL) ?
                            (token_valid[mc][ml] ? beta_q[mc][ml] : 32'b0) : 32'b0;
                        wire [31:0] mac_b = (state == S_QK_MUL) ?
                            qk_key_word :
                            (state == S_PV_MUL) ?
                            (token_valid[mc][ml] ? pv_value_word : 32'b0) : 32'b0;
                        wire [7:0] mac_status;
                        DW_fp_mult u_mac(
                            .a(mac_a), .b(mac_b), .rnd(3'b000),
                            .z(mac_product[mc][mb][mp][ml]), .status(mac_status));
                    end
                end
            end
        end
    endgenerate

    wire [31:0] compact_tree [0:CONTEXT_PARALLEL-1][0:BLOCK_PARALLEL-1]
        [0:MAX_TREE_LEVELS][0:TILE_SIZE-1][0:TILE_SIZE-1];
    wire [31:0] mac_stage_sum [0:CONTEXT_PARALLEL-1][0:BLOCK_PARALLEL-1]
        [1:MAX_TREE_LEVELS][0:TILE_SIZE-1][0:TILE_SIZE-1];
    genvar ctc, ctb, ctl, ctp, ctn;
    generate
        for (ctc = 0; ctc < CONTEXT_PARALLEL; ctc = ctc + 1) begin : g_comp_ctx
            for (ctb = 0; ctb < BLOCK_PARALLEL; ctb = ctb + 1) begin : g_comp_blk
                for (ctp = 0; ctp < TILE_SIZE; ctp = ctp + 1)
                    for (ctn = 0; ctn < TILE_SIZE; ctn = ctn + 1)
                        assign compact_tree[ctc][ctb][0][ctp][ctn] =
                            mac_product[ctc][ctb][ctp][ctn];
                for (ctl = 1; ctl <= MAX_TREE_LEVELS; ctl = ctl + 1) begin : g_comp_level
                    for (ctp = 0; ctp < TILE_SIZE; ctp = ctp + 1) begin : g_comp_row
                        for (ctn = 0; ctn < TILE_SIZE; ctn = ctn + 1) begin : g_comp_node
                            if (ctl <= TREE_LEVELS && ctn < (TILE_SIZE >> ctl)) begin : g_real
                                wire [7:0] compact_status;
                                DW_fp_addsub u_compact_add(
                                    .a(compact_tree[ctc][ctb][ctl-1][ctp][2*ctn]),
                                    .b(compact_tree[ctc][ctb][ctl-1][ctp][2*ctn+1]),
                                    .rnd(3'b000), .op(1'b0),
                                    .z(compact_tree[ctc][ctb][ctl][ctp][ctn]),
                                    .status(compact_status));
                            end else if (ctn == 0) begin : g_forward
                                assign compact_tree[ctc][ctb][ctl][ctp][ctn] =
                                    compact_tree[ctc][ctb][ctl-1][ctp][0];
                            end else begin : g_empty
                                assign compact_tree[ctc][ctb][ctl][ctp][ctn] = 32'b0;
                            end
                        end
                    end
                end
            end
        end
    endgenerate

    genvar mrl, mrp, mrn;
    generate
        for (ctc = 0; ctc < CONTEXT_PARALLEL; ctc = ctc + 1) begin : g_stage_ctx
            for (ctb = 0; ctb < BLOCK_PARALLEL; ctb = ctb + 1) begin : g_stage_blk
                for (mrl = 1; mrl <= MAX_TREE_LEVELS; mrl = mrl + 1) begin : g_stage_level
                    for (mrp = 0; mrp < TILE_SIZE; mrp = mrp + 1) begin : g_stage_row
                        for (mrn = 0; mrn < TILE_SIZE; mrn = mrn + 1) begin : g_stage_node
                            if (mrl <= TREE_LEVELS && mrn < (TILE_SIZE >> mrl)) begin : g_real
                                wire [7:0] stage_status;
                                DW_fp_addsub u_stage_add(
                                    .a(mac_tree_q[ctc][ctb][mrl-1][mrp][2*mrn]),
                                    .b(mac_tree_q[ctc][ctb][mrl-1][mrp][2*mrn+1]),
                                    .rnd(3'b000), .op(1'b0),
                                    .z(mac_stage_sum[ctc][ctb][mrl][mrp][mrn]),
                                    .status(stage_status));
                            end else if (mrn == 0) begin : g_forward
                                assign mac_stage_sum[ctc][ctb][mrl][mrp][mrn] =
                                    mac_tree_q[ctc][ctb][mrl-1][mrp][0];
                            end else begin : g_empty
                                assign mac_stage_sum[ctc][ctb][mrl][mrp][mrn] = 32'b0;
                            end
                        end
                    end
                end
            end
        end
    endgenerate

    wire [31:0] selected_mac_root [0:CONTEXT_PARALLEL-1][0:BLOCK_PARALLEL-1][0:TILE_SIZE-1];
    wire [31:0] qk_group_sum_wire [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    reg  [31:0] qk_group_sum_q [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    wire [31:0] qk_accum_sum [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    genvar src, srb, srp, qgc, qgp;
    generate
        for (src = 0; src < CONTEXT_PARALLEL; src = src + 1)
            for (srb = 0; srb < BLOCK_PARALLEL; srb = srb + 1)
                for (srp = 0; srp < TILE_SIZE; srp = srp + 1)
                    assign selected_mac_root[src][srb][srp] = PIPELINE_MAC ?
                        mac_tree_q[src][srb][MAX_TREE_LEVELS][srp][0] :
                        compact_tree[src][srb][MAX_TREE_LEVELS][srp][0];
        for (qgc = 0; qgc < CONTEXT_PARALLEL; qgc = qgc + 1) begin : g_qk_group
            for (qgp = 0; qgp < TILE_SIZE; qgp = qgp + 1) begin : g_qk_sum
                wire [7:0] group_status;
                DW_fp_addsub u_group_add(
                    .a(selected_mac_root[qgc][0][qgp]),
                    .b(selected_mac_root[qgc][1][qgp]),
                    .rnd(3'b000), .op(1'b0), .z(qk_group_sum_wire[qgc][qgp]),
                    .status(group_status));
                wire [7:0] accum_status;
                DW_fp_addsub u_accum_add(
                    .a(score_accum[qgc][qgp]), .b(qk_group_sum_q[qgc][qgp]),
                    .rnd(3'b000), .op(1'b0), .z(qk_accum_sum[qgc][qgp]),
                    .status(accum_status));
            end
        end
    endgenerate

    wire [31:0] sm_max_stage [0:CONTEXT_PARALLEL-1][0:MAX_TREE_LEVELS][0:TILE_SIZE-1];
    genvar smc, smx, smy;
    generate
        for (smc = 0; smc < CONTEXT_PARALLEL; smc = smc + 1) begin : g_max_ctx
            for (smx = 0; smx < TILE_SIZE; smx = smx + 1)
                assign sm_max_stage[smc][0][smx] = token_valid[smc][smx] ?
                    score_final[smc][smx] : FP_NEG_INF;
            for (smy = 1; smy <= MAX_TREE_LEVELS; smy = smy + 1) begin : g_max_level
                for (smx = 0; smx < TILE_SIZE; smx = smx + 1) begin : g_max_node
                    if (smy <= TREE_LEVELS && smx < (TILE_SIZE >> smy)) begin : g_real
                        assign sm_max_stage[smc][smy][smx] =
                            fp_max(sm_max_stage[smc][smy-1][2*smx],
                                   sm_max_stage[smc][smy-1][2*smx+1]);
                    end else if (smx == 0) begin : g_forward
                        assign sm_max_stage[smc][smy][smx] =
                            sm_max_stage[smc][smy-1][0];
                    end else begin : g_empty
                        assign sm_max_stage[smc][smy][smx] = FP_NEG_INF;
                    end
                end
            end
        end
    endgenerate

    // For a 32-token tile, reduce eight 4-token maxima in two additional
    // balanced stages: 8->4, then 4->1.  No cycle has more than two compare
    // levels, and all independent branches compare concurrently.
    wire [31:0] sm_group_final_stage [0:CONTEXT_PARALLEL-1]
        [0:MAX_REMAIN_LEVELS][0:MAX_REMAIN_GROUPS-1];
    genvar smrc, smrl, smrx;
    generate
        for (smrc = 0; smrc < CONTEXT_PARALLEL; smrc = smrc + 1) begin : g_max_group_final
            for (smrx = 0; smrx < MAX_REMAIN_GROUPS; smrx = smrx + 1)
                assign sm_group_final_stage[smrc][0][smrx] =
                    sm_max_tree_q[smrc][1][smrx];
            for (smrl = 1; smrl <= MAX_REMAIN_LEVELS; smrl = smrl + 1) begin : g_level
                for (smrx = 0; smrx < MAX_REMAIN_GROUPS; smrx = smrx + 1) begin : g_node
                    if (smrx < (MAX_REMAIN_GROUPS >> smrl)) begin : g_real
                        assign sm_group_final_stage[smrc][smrl][smrx] =
                            fp_max(sm_group_final_stage[smrc][smrl-1][2*smrx],
                                   sm_group_final_stage[smrc][smrl-1][2*smrx+1]);
                    end else if (smrx == 0) begin : g_forward
                        assign sm_group_final_stage[smrc][smrl][smrx] =
                            sm_group_final_stage[smrc][smrl-1][0];
                    end else begin : g_empty
                        assign sm_group_final_stage[smrc][smrl][smrx] = FP_NEG_INF;
                    end
                end
            end
        end
    endgenerate

    wire [31:0] beta_diff [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    wire [31:0] beta_exp [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    wire [31:0] sm_sum_input [0:CONTEXT_PARALLEL-1][0:TILE_SIZE-1];
    genvar bec, bet;
    generate
        for (bec = 0; bec < CONTEXT_PARALLEL; bec = bec + 1) begin : g_beta_ctx
            for (bet = 0; bet < TILE_SIZE; bet = bet + 1) begin : g_beta
                wire [7:0] diff_status, exp_status;
                DW_fp_addsub u_beta_diff(
                    .a(score_final[bec][bet]), .b(local_max_q[bec]),
                    .rnd(3'b000), .op(1'b1), .z(beta_diff[bec][bet]),
                    .status(diff_status));
                DW_fp_exp u_beta_exp(
                    .a(beta_diff_q[bec][bet]), .z(beta_exp[bec][bet]),
                    .status(exp_status));
                assign sm_sum_input[bec][bet] = beta_q[bec][bet];
            end
        end
    endgenerate

    wire [31:0] sm_sum_stage [0:CONTEXT_PARALLEL-1][0:MAX_TREE_LEVELS][0:TILE_SIZE-1];
    genvar ssc, ssx, ssy;
    generate
        for (ssc = 0; ssc < CONTEXT_PARALLEL; ssc = ssc + 1) begin : g_sum_ctx
            for (ssx = 0; ssx < TILE_SIZE; ssx = ssx + 1)
                assign sm_sum_stage[ssc][0][ssx] = sm_sum_input[ssc][ssx];
            for (ssy = 1; ssy <= MAX_TREE_LEVELS; ssy = ssy + 1) begin : g_sum_level
                for (ssx = 0; ssx < TILE_SIZE; ssx = ssx + 1) begin : g_sum_node
                    if (ssy <= TREE_LEVELS && ssx < (TILE_SIZE >> ssy)) begin : g_real
                        wire [7:0] sum_status;
                        DW_fp_addsub u_sum_add(
                            .a(PIPELINE_SOFTMAX ? sm_sum_tree_q[ssc][ssy-1][2*ssx] :
                               sm_sum_stage[ssc][ssy-1][2*ssx]),
                            .b(PIPELINE_SOFTMAX ? sm_sum_tree_q[ssc][ssy-1][2*ssx+1] :
                               sm_sum_stage[ssc][ssy-1][2*ssx+1]),
                            .rnd(3'b000), .op(1'b0),
                            .z(sm_sum_stage[ssc][ssy][ssx]), .status(sum_status));
                    end else if (ssx == 0) begin : g_forward
                        assign sm_sum_stage[ssc][ssy][ssx] = PIPELINE_SOFTMAX ?
                            sm_sum_tree_q[ssc][ssy-1][0] : sm_sum_stage[ssc][ssy-1][0];
                    end else begin : g_empty
                        assign sm_sum_stage[ssc][ssy][ssx] = 32'b0;
                    end
                end
            end
        end
    endgenerate

    wire [31:0] sm_sum_group_stage [0:CONTEXT_PARALLEL-1]
        [0:MAX_GROUP_LEVELS][0:TILE_SIZE-1];
    genvar sfgc, sfgl, sfgx;
    generate
        for (sfgc = 0; sfgc < CONTEXT_PARALLEL; sfgc = sfgc + 1) begin : g_sum_group_first
            for (sfgx = 0; sfgx < TILE_SIZE; sfgx = sfgx + 1)
                assign sm_sum_group_stage[sfgc][0][sfgx] = beta_q[sfgc][sfgx];
            for (sfgl = 1; sfgl <= MAX_GROUP_LEVELS; sfgl = sfgl + 1) begin : g_level
                for (sfgx = 0; sfgx < TILE_SIZE; sfgx = sfgx + 1) begin : g_node
                    if (sfgx < (TILE_SIZE >> sfgl)) begin : g_real
                        wire [7:0] group_sum_status;
                        DW_fp_addsub u_group_sum(
                            .a(sm_sum_group_stage[sfgc][sfgl-1][2*sfgx]),
                            .b(sm_sum_group_stage[sfgc][sfgl-1][2*sfgx+1]),
                            .rnd(3'b000), .op(1'b0),
                            .z(sm_sum_group_stage[sfgc][sfgl][sfgx]),
                            .status(group_sum_status));
                    end else if (sfgx == 0) begin : g_forward
                        assign sm_sum_group_stage[sfgc][sfgl][sfgx] =
                            sm_sum_group_stage[sfgc][sfgl-1][0];
                    end else begin : g_empty
                        assign sm_sum_group_stage[sfgc][sfgl][sfgx] = FP_ZERO;
                    end
                end
            end
        end
    endgenerate

    wire [31:0] sm_sum_group_pair [0:CONTEXT_PARALLEL-1]
        [0:MAX_SUM_REMAIN_GROUPS-1];
    genvar sgpc, sgpx;
    generate
        for (sgpc = 0; sgpc < CONTEXT_PARALLEL; sgpc = sgpc + 1) begin : g_sum_group_pair
            for (sgpx = 0; sgpx < MAX_SUM_REMAIN_GROUPS; sgpx = sgpx + 1) begin : g_pair
                wire [7:0] pair_sum_status;
                DW_fp_addsub u_pair_sum(
                    .a(sm_sum_tree_q[sgpc][0][2*sgpx]),
                    .b(sm_sum_tree_q[sgpc][0][2*sgpx+1]),
                    .rnd(3'b000), .op(1'b0), .z(sm_sum_group_pair[sgpc][sgpx]),
                    .status(pair_sum_status));
            end
        end
    endgenerate

    wire [31:0] sm_sum_group_final_stage [0:CONTEXT_PARALLEL-1]
        [0:MAX_SUM_REMAIN_LEVELS][0:MAX_SUM_REMAIN_GROUPS-1];
    genvar sfrc, sfrl, sfrx;
    generate
        for (sfrc = 0; sfrc < CONTEXT_PARALLEL; sfrc = sfrc + 1) begin : g_sum_group_final
            for (sfrx = 0; sfrx < MAX_SUM_REMAIN_GROUPS; sfrx = sfrx + 1)
                assign sm_sum_group_final_stage[sfrc][0][sfrx] =
                    sm_sum_tree_q[sfrc][1][sfrx];
            for (sfrl = 1; sfrl <= MAX_SUM_REMAIN_LEVELS; sfrl = sfrl + 1) begin : g_level
                for (sfrx = 0; sfrx < MAX_SUM_REMAIN_GROUPS; sfrx = sfrx + 1) begin : g_node
                    if (sfrx < (MAX_SUM_REMAIN_GROUPS >> sfrl)) begin : g_real
                        wire [7:0] final_sum_status;
                        DW_fp_addsub u_final_sum(
                            .a(sm_sum_group_final_stage[sfrc][sfrl-1][2*sfrx]),
                            .b(sm_sum_group_final_stage[sfrc][sfrl-1][2*sfrx+1]),
                            .rnd(3'b000), .op(1'b0),
                            .z(sm_sum_group_final_stage[sfrc][sfrl][sfrx]),
                            .status(final_sum_status));
                    end else if (sfrx == 0) begin : g_forward
                        assign sm_sum_group_final_stage[sfrc][sfrl][sfrx] =
                            sm_sum_group_final_stage[sfrc][sfrl-1][0];
                    end else begin : g_empty
                        assign sm_sum_group_final_stage[sfrc][sfrl][sfrx] = FP_ZERO;
                    end
                end
            end
        end
    endgenerate

    wire [31:0] selected_pv_root [0:CONTEXT_PARALLEL-1][0:BLOCK_PARALLEL-1][0:TILE_SIZE-1];
    genvar pvc, pvb, pvp;
    generate
        for (pvc = 0; pvc < CONTEXT_PARALLEL; pvc = pvc + 1)
            for (pvb = 0; pvb < BLOCK_PARALLEL; pvb = pvb + 1)
                for (pvp = 0; pvp < TILE_SIZE; pvp = pvp + 1)
                    assign selected_pv_root[pvc][pvb][pvp] = PIPELINE_MAC ?
                        selected_mac_root[pvc][pvb][pvp] : pv_root_q[pvc][pvb][pvp];
    endgenerate

    wire [31:0] pair_max_wire = fp_max(
        tile_valid_q[0] ? local_max_q[0] : FP_NEG_INF,
        tile_valid_q[1] ? local_max_q[1] : FP_NEG_INF);
    wire [31:0] new_max_wire = fp_max(
        have_context_q ? running_max_q : FP_NEG_INF, pair_max_q);
    wire [31:0] pair_alpha_diff [0:CONTEXT_PARALLEL-1];
    wire [31:0] pair_alpha_exp [0:CONTEXT_PARALLEL-1];
    wire [31:0] global_alpha_diff, global_alpha_exp, pair_beta_diff, pair_beta_exp;
    genvar pac;
    generate
        for (pac = 0; pac < CONTEXT_PARALLEL; pac = pac + 1) begin : g_pair_alpha
            wire [7:0] ad_status, ae_status;
            DW_fp_addsub u_pair_alpha_diff(
                .a(local_max_q[pac]), .b(pair_max_q), .rnd(3'b000), .op(1'b1),
                .z(pair_alpha_diff[pac]), .status(ad_status));
            DW_fp_exp u_pair_alpha_exp(
                .a(pair_alpha_diff_q[pac]), .z(pair_alpha_exp[pac]), .status(ae_status));
        end
        wire [7:0] gad_status, gae_status, gbd_status, gbe_status;
        DW_fp_addsub u_global_alpha_diff(
            .a(have_context_q ? running_max_q : pair_max_q), .b(new_max_q),
            .rnd(3'b000), .op(1'b1), .z(global_alpha_diff), .status(gad_status));
        DW_fp_exp u_global_alpha_exp(
            .a(global_alpha_diff_q), .z(global_alpha_exp), .status(gae_status));
        DW_fp_addsub u_pair_beta_diff(
            .a(pair_max_q), .b(new_max_q), .rnd(3'b000), .op(1'b1),
            .z(pair_beta_diff), .status(gbd_status));
        DW_fp_exp u_pair_beta_exp(
            .a(pair_beta_diff_q), .z(pair_beta_exp), .status(gbe_status));
    endgenerate

    wire [31:0] pair_sum_prod_wire [0:CONTEXT_PARALLEL-1];
    wire [31:0] pair_z_prod_wire [0:CONTEXT_PARALLEL-1][0:HEAD_DIM-1];
    genvar psc, psd;
    generate
        for (psc = 0; psc < CONTEXT_PARALLEL; psc = psc + 1) begin : g_pair_prod
            wire [7:0] sum_prod_status;
            DW_fp_mult u_pair_sum_prod(
                .a(pair_alpha_q[psc]), .b(local_sum_q[psc]), .rnd(3'b000),
                .z(pair_sum_prod_wire[psc]), .status(sum_prod_status));
            for (psd = 0; psd < HEAD_DIM; psd = psd + 1) begin : g_pair_z_prod
                wire [7:0] z_prod_status;
                DW_fp_mult u_pair_z_prod(
                    .a(pair_alpha_q[psc]), .b(local_z_q[psc][psd]), .rnd(3'b000),
                    .z(pair_z_prod_wire[psc][psd]), .status(z_prod_status));
            end
        end
    endgenerate
    wire [31:0] pair_sum_wire;
    wire [31:0] pair_z_sum_wire [0:HEAD_DIM-1];
    wire [7:0] pair_sum_status;
    DW_fp_addsub u_pair_sum_add(
        .a(pair_sum_prod_q[0]), .b(pair_sum_prod_q[1]), .rnd(3'b000), .op(1'b0),
        .z(pair_sum_wire), .status(pair_sum_status));
    genvar pzd;
    generate
        for (pzd = 0; pzd < HEAD_DIM; pzd = pzd + 1) begin : g_pair_z_add
            wire [7:0] z_sum_status;
            DW_fp_addsub u_pair_z_add(
                .a(pair_z_prod_q[0][pzd]), .b(pair_z_prod_q[1][pzd]),
                .rnd(3'b000), .op(1'b0), .z(pair_z_sum_wire[pzd]),
                .status(z_sum_status));
        end
    endgenerate

    wire [31:0] global_sum_product_wire, pair_sum_global_wire, sum_update_wire;
    wire [7:0] gsm_status, gsp_status, sup_status;
    DW_fp_mult u_global_sum_product(
        .a(global_alpha_q), .b(running_sum_q), .rnd(3'b000),
        .z(global_sum_product_wire), .status(gsm_status));
    DW_fp_mult u_pair_sum_global_product(
        .a(pair_beta_q), .b(pair_sum_q), .rnd(3'b000),
        .z(pair_sum_global_wire), .status(gsp_status));
    DW_fp_addsub u_sum_update(
        .a(global_sum_prod_q), .b(pair_sum_global_prod_q), .rnd(3'b000),
        .op(1'b0), .z(sum_update_wire), .status(sup_status));

    wire [31:0] global_z_product_wire [0:HEAD_DIM-1];
    wire [31:0] pair_z_product_wire [0:HEAD_DIM-1];
    genvar gzd;
    generate
        for (gzd = 0; gzd < HEAD_DIM; gzd = gzd + 1) begin : g_global_z
            wire [7:0] gp_status, pp_status;
            DW_fp_mult u_global_z_product(
                .a(global_alpha_q), .b(output_accum[gzd]), .rnd(3'b000),
                .z(global_z_product_wire[gzd]), .status(gp_status));
            DW_fp_mult u_pair_z_global_product(
                .a(pair_beta_q), .b(pair_z_q[gzd]), .rnd(3'b000),
                .z(pair_z_product_wire[gzd]), .status(pp_status));
        end
    endgenerate
    reg [31:0] global_z_prod_q [0:HEAD_DIM-1];
    reg [31:0] pair_z_global_prod_q [0:HEAD_DIM-1];
    wire [31:0] output_update_wire [0:HEAD_DIM-1];
    genvar ou;
    generate
        for (ou = 0; ou < HEAD_DIM; ou = ou + 1) begin : g_output_update
            wire [7:0] out_status;
            DW_fp_addsub u_output_update(
                .a(global_z_prod_q[ou]), .b(pair_z_global_prod_q[ou]),
                .rnd(3'b000), .op(1'b0), .z(output_update_wire[ou]),
                .status(out_status));
        end
    endgenerate

    wire [31:0] inverse_wire;
    wire [7:0] inverse_status;
    DW_fp_div u_inverse(
        .a(FP_ONE), .b(running_sum_q), .rnd(3'b000),
        .z(inverse_wire), .status(inverse_status));
    wire [31:0] final_scale_wire [0:HEAD_DIM-1];
    genvar fs;
    generate
        for (fs = 0; fs < HEAD_DIM; fs = fs + 1) begin : g_final_scale
            wire [7:0] final_status;
            DW_fp_mult u_final_scale(
                .a(output_accum[fs]), .b(inverse_sum_q), .rnd(3'b000),
                .z(final_scale_wire[fs]), .status(final_status));
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            context_len_q <= 0;
            processed_keys_q <= 0;
            qk_group_q <= 0;
            pv_group_q <= 0;
            load_lane_q <= 0;
            tile_valid_q <= 0;
            query_q <= 0;
            query_scaled_q <= 0;
            result_q <= 0;
            running_max_q <= FP_NEG_INF;
            running_sum_q <= FP_ZERO;
            have_context_q <= 0;
            pair_max_q <= FP_NEG_INF;
            new_max_q <= FP_NEG_INF;
            global_alpha_diff_q <= FP_ZERO;
            pair_beta_diff_q <= FP_ZERO;
            pair_beta_q <= FP_ZERO;
            global_alpha_q <= FP_ZERO;
            inverse_sum_q <= FP_ZERO;
            o_error <= 0;
            o_cycle_count <= 0;
            for (init_c = 0; init_c < CONTEXT_PARALLEL; init_c = init_c + 1) begin
                local_max_q[init_c] <= FP_NEG_INF;
                token_valid_q[init_c] <= {TILE_SIZE{1'b0}};
                pair_alpha_diff_q[init_c] <= FP_ZERO;
                local_sum_q[init_c] <= FP_ZERO;
                for (init_i = 0; init_i < TILE_SIZE; init_i = init_i + 1) begin
                    score_accum[init_c][init_i] <= FP_ZERO;
                    score_final[init_c][init_i] <= FP_ZERO;
                    beta_q[init_c][init_i] <= FP_ZERO;
                    beta_diff_q[init_c][init_i] <= FP_ZERO;
                end
                for (init_i = 0; init_i < HEAD_DIM; init_i = init_i + 1)
                    local_z_q[init_c][init_i] <= FP_ZERO;
                for (init_b = 0; init_b < BLOCK_PARALLEL; init_b = init_b + 1)
                    for (init_i = 0; init_i < TILE_SIZE; init_i = init_i + 1)
                        pv_root_q[init_c][init_b][init_i] <= FP_ZERO;
            end
            for (init_i = 0; init_i < HEAD_DIM; init_i = init_i + 1) begin
                output_accum[init_i] <= FP_ZERO;
                pair_z_q[init_i] <= FP_ZERO;
                global_z_prod_q[init_i] <= FP_ZERO;
                pair_z_global_prod_q[init_i] <= FP_ZERO;
            end
            for (init_c = 0; init_c < CONTEXT_PARALLEL; init_c = init_c + 1)
                for (init_b = 0; init_b < BLOCK_PARALLEL; init_b = init_b + 1)
                    for (init_l = 0; init_l <= MAX_TREE_LEVELS; init_l = init_l + 1)
                        for (init_i = 0; init_i < TILE_SIZE; init_i = init_i + 1)
                            for (init_j = 0; init_j < TILE_SIZE; init_j = init_j + 1)
                                mac_tree_q[init_c][init_b][init_l][init_i][init_j] <= FP_ZERO;
            for (init_c = 0; init_c < CONTEXT_PARALLEL; init_c = init_c + 1)
                for (init_l = 0; init_l <= MAX_TREE_LEVELS; init_l = init_l + 1)
                    for (init_i = 0; init_i < TILE_SIZE; init_i = init_i + 1) begin
                        sm_max_tree_q[init_c][init_l][init_i] <= FP_NEG_INF;
                        sm_sum_tree_q[init_c][init_l][init_i] <= FP_ZERO;
                    end
        end else begin
            if (state != S_IDLE && state != S_RESPONSE)
                o_cycle_count <= o_cycle_count + 1'b1;
            case (state)
                S_IDLE: begin
                    if (i_cmd_valid) begin
                        context_len_q <= i_context_len;
                        processed_keys_q <= 0;
                        qk_group_q <= 0;
                        pv_group_q <= 0;
                        load_lane_q <= 0;
                        tile_valid_q <= 0;
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            token_valid_q[c] <= {TILE_SIZE{1'b0}};
                        have_context_q <= 0;
                        running_max_q <= FP_NEG_INF;
                        running_sum_q <= FP_ZERO;
                        result_q <= 0;
                        o_error <= (i_context_len == 0 || i_context_len > MAX_CONTEXT);
                        o_cycle_count <= 0;
                        query_q <= i_query;
                        for (p = 0; p < HEAD_DIM; p = p + 1)
                            output_accum[p] <= FP_ZERO;
                        if (i_context_len == 0 || i_context_len > MAX_CONTEXT)
                            state <= S_RESPONSE;
                        else state <= S_QUERY_SCALE;
                    end
                end
                S_QUERY_SCALE: begin
                    for (p = 0; p < HEAD_DIM; p = p + 1)
                        query_scaled_q[p*32 +: 32] <= PRE_SCALE_QUERY ?
                            query_scale_wire[p] : query_q[p*32 +: 32];
                    state <= S_WAIT_TILE;
                end
                S_WAIT_TILE: begin
                    if (i_kv_valid) begin
                        key_q[load_lane_q] <= i_key_tile;
                        value_q[load_lane_q] <= i_value_tile;
                        tile_valid_q[load_lane_q] <= 1'b1;
                        for (p = 0; p < TILE_SIZE; p = p + 1)
                            token_valid_q[load_lane_q][p] <=
                                (processed_keys_q + load_lane_q*TILE_SIZE + p < context_len_q);
                        if (load_lane_q == CONTEXT_PARALLEL-1 ||
                            processed_keys_q + TILE_SIZE >= context_len_q) begin
                            if (load_lane_q == 0) tile_valid_q[1] <= 1'b0;
                            qk_group_q <= 0;
                            for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                                for (p = 0; p < TILE_SIZE; p = p + 1)
                                    score_accum[c][p] <= FP_ZERO;
                            state <= S_QK_MUL;
                        end else load_lane_q <= load_lane_q + 1'b1;
                    end
                end
                S_QK_MUL: begin
                    if (PIPELINE_MAC) begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (b = 0; b < BLOCK_PARALLEL; b = b + 1)
                                for (p = 0; p < TILE_SIZE; p = p + 1)
                                    for (l = 0; l < TILE_SIZE; l = l + 1)
                                        mac_tree_q[c][b][0][p][l] <= mac_product[c][b][p][l];
                        state <= S_QK_R1;
                    end else begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (p = 0; p < TILE_SIZE; p = p + 1)
                                qk_group_sum_q[c][p] <= qk_group_sum_wire[c][p];
                        state <= S_QK_ACC;
                    end
                end
                S_QK_R1, S_QK_R2, S_QK_R3, S_QK_R4, S_QK_R5: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        for (b = 0; b < BLOCK_PARALLEL; b = b + 1)
                            for (p = 0; p < TILE_SIZE; p = p + 1)
                                for (l = 0; l < TILE_SIZE; l = l + 1)
                                    mac_tree_q[c][b][state-S_QK_R1+1][p][l] <=
                                        mac_stage_sum[c][b][state-S_QK_R1+1][p][l];
                    if (state == S_QK_R5) state <= S_QK_GROUP;
                    else state <= state + 1'b1;
                end
                S_QK_GROUP: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        for (p = 0; p < TILE_SIZE; p = p + 1)
                            qk_group_sum_q[c][p] <= qk_group_sum_wire[c][p];
                    state <= S_QK_ACC;
                end
                S_QK_ACC: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1) begin
                        for (p = 0; p < TILE_SIZE; p = p + 1) begin
                            score_accum[c][p] <= qk_accum_sum[c][p];
                            if (qk_group_q == QK_GROUPS-1)
                                score_final[c][p] <= qk_accum_sum[c][p];
                        end
                    end
                    if (qk_group_q == QK_GROUPS-1) state <= S_SM_MAX;
                    else begin
                        qk_group_q <= qk_group_q + 1'b1;
                        state <= S_QK_MUL;
                    end
                end
                S_SM_MAX: begin
                    if (PIPELINE_SOFTMAX) begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (p = 0; p < MAX_GROUPS; p = p + 1)
                                sm_max_tree_q[c][0][p] <=
                                    sm_max_stage[c][MAX_GROUP_LEVELS][p];
                        if (MAX_GROUPS == 1) begin
                            for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                                local_max_q[c] <= sm_max_stage[c][TREE_LEVELS][0];
                            state <= S_SM_DIFF;
                        end else state <= S_SM_MAX_R1;
                    end else begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            local_max_q[c] <= sm_max_stage[c][MAX_TREE_LEVELS][0];
                        state <= S_SM_DIFF;
                    end
                end
                S_SM_MAX_R1: begin
                    if (MAX_GROUPS == 2) begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            local_max_q[c] <= fp_max(sm_max_tree_q[c][0][0],
                                                     sm_max_tree_q[c][0][1]);
                        state <= S_SM_DIFF;
                    end else begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (p = 0; p < MAX_REMAIN_GROUPS; p = p + 1)
                                sm_max_tree_q[c][1][p] <=
                                    fp_max(sm_max_tree_q[c][0][2*p],
                                           sm_max_tree_q[c][0][2*p+1]);
                        state <= S_SM_MAX_R2;
                    end
                end
                S_SM_MAX_R2: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        local_max_q[c] <=
                            sm_group_final_stage[c][MAX_REMAIN_LEVELS][0];
                    state <= S_SM_DIFF;
                end
                S_SM_DIFF: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        for (p = 0; p < TILE_SIZE; p = p + 1)
                            beta_diff_q[c][p] <= beta_diff[c][p];
                    state <= S_SM_EXP;
                end
                S_SM_EXP: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        for (p = 0; p < TILE_SIZE; p = p + 1)
                            beta_q[c][p] <= token_valid[c][p] ? beta_exp[c][p] : FP_ZERO;
                    state <= S_SM_SUM;
                end
                S_SM_SUM: begin
                    if (PIPELINE_SOFTMAX) begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (p = 0; p < MAX_GROUPS; p = p + 1)
                                sm_sum_tree_q[c][0][p] <=
                                    sm_sum_group_stage[c][MAX_GROUP_LEVELS][p];
                        if (MAX_GROUPS == 1) begin
                            for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                                local_sum_q[c] <=
                                    sm_sum_group_stage[c][MAX_GROUP_LEVELS][0];
                            pv_group_q <= 0;
                            for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                                for (p = 0; p < HEAD_DIM; p = p + 1)
                                    local_z_q[c][p] <= FP_ZERO;
                            state <= S_PV_MUL;
                        end else state <= S_SM_SUM_R1;
                    end else begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            local_sum_q[c] <= sm_sum_stage[c][MAX_TREE_LEVELS][0];
                        pv_group_q <= 0;
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (p = 0; p < HEAD_DIM; p = p + 1)
                                local_z_q[c][p] <= FP_ZERO;
                        state <= S_PV_MUL;
                    end
                end
                S_SM_SUM_R1: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        for (p = 0; p < MAX_SUM_REMAIN_GROUPS; p = p + 1)
                            sm_sum_tree_q[c][1][p] <=
                                sm_sum_group_pair[c][p];
                    if (MAX_SUM_REMAIN_GROUPS == 1) begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            local_sum_q[c] <= sm_sum_group_pair[c][0];
                        pv_group_q <= 0;
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (p = 0; p < HEAD_DIM; p = p + 1)
                                local_z_q[c][p] <= FP_ZERO;
                        state <= S_PV_MUL;
                    end else state <= S_SM_SUM_R2;
                end
                S_SM_SUM_R2: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        local_sum_q[c] <=
                            sm_sum_group_final_stage[c][MAX_SUM_REMAIN_LEVELS][0];
                    pv_group_q <= 0;
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        for (p = 0; p < HEAD_DIM; p = p + 1)
                            local_z_q[c][p] <= FP_ZERO;
                    state <= S_PV_MUL;
                end
                S_PV_MUL: begin
                    if (PIPELINE_MAC) begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (b = 0; b < BLOCK_PARALLEL; b = b + 1)
                                for (p = 0; p < TILE_SIZE; p = p + 1)
                                    for (l = 0; l < TILE_SIZE; l = l + 1)
                                        mac_tree_q[c][b][0][p][l] <= mac_product[c][b][p][l];
                        state <= S_PV_R1;
                    end else begin
                        for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                            for (b = 0; b < BLOCK_PARALLEL; b = b + 1)
                                for (p = 0; p < TILE_SIZE; p = p + 1)
                                    pv_root_q[c][b][p] <= compact_tree[c][b][MAX_TREE_LEVELS][p][0];
                        state <= S_PV_ACC;
                    end
                end
                S_PV_R1, S_PV_R2, S_PV_R3, S_PV_R4, S_PV_R5: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        for (b = 0; b < BLOCK_PARALLEL; b = b + 1)
                            for (p = 0; p < TILE_SIZE; p = p + 1)
                                for (l = 0; l < TILE_SIZE; l = l + 1)
                                    mac_tree_q[c][b][state-S_PV_R1+1][p][l] <=
                                        mac_stage_sum[c][b][state-S_PV_R1+1][p][l];
                    if (state == S_PV_R5) state <= S_PV_ACC;
                    else state <= state + 1'b1;
                end
                S_PV_ACC: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        for (b = 0; b < BLOCK_PARALLEL; b = b + 1)
                            for (p = 0; p < TILE_SIZE; p = p + 1)
                                local_z_q[c][pv_group_q*BLOCK_PARALLEL*TILE_SIZE+
                                    b*TILE_SIZE+p] <= selected_pv_root[c][b][p];
                    if (pv_group_q == OUT_GROUPS-1) state <= S_PAIR_MAX;
                    else begin
                        pv_group_q <= pv_group_q + 1'b1;
                        state <= S_PV_MUL;
                    end
                end
                S_PAIR_MAX: begin
                    pair_max_q <= pair_max_wire;
                    new_max_q <= fp_max(have_context_q ? running_max_q : FP_NEG_INF,
                                        pair_max_wire);
                    state <= S_PAIR_DIFF;
                end
                S_PAIR_DIFF: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        pair_alpha_diff_q[c] <= pair_alpha_diff[c];
                    global_alpha_diff_q <= global_alpha_diff;
                    pair_beta_diff_q <= pair_beta_diff;
                    state <= S_PAIR_EXP;
                end
                S_PAIR_EXP: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1)
                        pair_alpha_q[c] <= tile_valid_q[c] ? pair_alpha_exp[c] : FP_ZERO;
                    global_alpha_q <= have_context_q ? global_alpha_exp : FP_ZERO;
                    pair_beta_q <= pair_beta_exp;
                    state <= S_PAIR_SCALE;
                end
                S_PAIR_SCALE: begin
                    for (c = 0; c < CONTEXT_PARALLEL; c = c + 1) begin
                        pair_sum_prod_q[c] <= pair_sum_prod_wire[c];
                        for (p = 0; p < HEAD_DIM; p = p + 1)
                            pair_z_prod_q[c][p] <= pair_z_prod_wire[c][p];
                    end
                    state <= S_PAIR_SUM;
                end
                S_PAIR_SUM: begin
                    pair_sum_q <= pair_sum_wire;
                    for (p = 0; p < HEAD_DIM; p = p + 1)
                        pair_z_q[p] <= pair_z_sum_wire[p];
                    state <= S_GLOBAL_SCALE;
                end
                S_GLOBAL_SCALE: begin
                    global_sum_prod_q <= global_sum_product_wire;
                    pair_sum_global_prod_q <= pair_sum_global_wire;
                    for (p = 0; p < HEAD_DIM; p = p + 1) begin
                        global_z_prod_q[p] <= global_z_product_wire[p];
                        pair_z_global_prod_q[p] <= pair_z_product_wire[p];
                    end
                    state <= S_GLOBAL_UPDATE;
                end
                S_GLOBAL_UPDATE: begin
                    running_max_q <= new_max_q;
                    running_sum_q <= sum_update_wire;
                    have_context_q <= 1'b1;
                    for (p = 0; p < HEAD_DIM; p = p + 1)
                        output_accum[p] <= output_update_wire[p];
                    if (processed_keys_q + CONTEXT_PARALLEL*TILE_SIZE >= context_len_q)
                        state <= S_FINAL_INV;
                    else begin
                        processed_keys_q <= processed_keys_q + CONTEXT_PARALLEL*TILE_SIZE;
                        load_lane_q <= 0;
                        tile_valid_q <= 0;
                        state <= S_WAIT_TILE;
                    end
                end
                S_FINAL_INV: begin
                    inverse_sum_q <= inverse_wire;
                    state <= S_FINAL_SCALE;
                end
                S_FINAL_SCALE: begin
                    for (d = 0; d < HEAD_DIM; d = d + 1)
                        result_q[d*32 +: 32] <= final_scale_wire[d];
                    state <= S_RESPONSE;
                end
                S_RESPONSE: begin
                    if (i_ready) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
