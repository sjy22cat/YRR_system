`timescale 1ns/1ps

// Experimental single-column, single-tile fused attention datapath.
//
// The tile has TILE_SIZE PEs, each with TILE_SIZE FP32 MAC lanes.  The same
// multiplier/reduction tile is time-multiplexed between QK^T, online-softmax
// normalization, and PV.  HEAD_DIM is traversed in TILE_SIZE-wide slices.
// This module handles one query row and accepts K/V in TILE_SIZE-token blocks.
// It is standalone and is not connected to the main YRR top level.
module attention_confusion_single_column #(
    parameter integer HEAD_DIM    = 128,
    parameter integer TILE_SIZE   = 32,
    parameter integer MAX_CONTEXT = 1152,
    parameter [31:0] SCALE_BITS   = 32'h3db504f3 // 1/sqrt(128), Qwen3 head_dim=128
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                  i_cmd_valid,
    output wire                  o_cmd_ready,
    input  wire [10:0]           i_context_len,
    input  wire [HEAD_DIM*32-1:0] i_query,

    input  wire                                      i_kv_valid,
    output wire                                      o_kv_ready,
    input  wire [TILE_SIZE*HEAD_DIM*32-1:0]          i_key_tile,
    input  wire [TILE_SIZE*HEAD_DIM*32-1:0]          i_value_tile,

    output wire                  o_valid,
    input  wire                  i_ready,
    output wire [HEAD_DIM*32-1:0] o_result,
    output reg                   o_error,
    output reg  [31:0]           o_cycle_count
);
    localparam integer QK_CHUNKS = HEAD_DIM / TILE_SIZE;
    localparam integer OUT_CHUNKS = HEAD_DIM / TILE_SIZE;

    localparam [3:0] S_IDLE        = 4'd0;
    localparam [3:0] S_WAIT_TILE   = 4'd1;
    localparam [3:0] S_DOT         = 4'd2;
    localparam [3:0] S_SCALE_QK    = 4'd3;
    localparam [3:0] S_ONLINE      = 4'd4;
    localparam [3:0] S_PV_SUM      = 4'd5;
    localparam [3:0] S_PV_SCALE    = 4'd6;
    localparam [3:0] S_FINAL_INV   = 4'd7;
    localparam [3:0] S_FINAL_SCALE = 4'd8;
    localparam [3:0] S_RESPONSE    = 4'd9;

    reg [3:0] state;
    reg [HEAD_DIM*32-1:0] query_q;
    reg [TILE_SIZE*HEAD_DIM*32-1:0] key_q, value_q;
    reg [HEAD_DIM*32-1:0] result_q;
    reg [31:0] output_accum [0:HEAD_DIM-1];
    reg [31:0] score_accum [0:TILE_SIZE-1];
    reg [31:0] score_final [0:TILE_SIZE-1];
    reg [31:0] beta_q [0:TILE_SIZE-1];
    reg [31:0] pv_partial [0:HEAD_DIM-1];

    reg [31:0] running_max_q;
    reg [31:0] running_sum_q;
    reg [31:0] alpha_q;
    reg [31:0] inverse_sum_q;
    reg        have_context_q;
    reg [10:0] context_len_q;
    reg [10:0] processed_keys_q;
    reg [10:0] tile_count_q;
    reg [$clog2(QK_CHUNKS+1)-1:0] qk_chunk_q;
    reg [$clog2(OUT_CHUNKS+1)-1:0] out_chunk_q;

    integer remaining_i;
    reg [10:0] next_tile_count;

    function automatic [31:0] fp_max(input [31:0] a, input [31:0] b);
        begin
            if (a[31] != b[31])
                fp_max = a[31] ? b : a;
            else if (a[31])
                fp_max = (a[30:0] < b[30:0]) ? a : b;
            else
                fp_max = (a[30:0] > b[30:0]) ? a : b;
        end
    endfunction

    function automatic fp_gt(input [31:0] a, input [31:0] b);
        begin
            fp_gt = (fp_max(a, b) == a) && (a != b);
        end
    endfunction

    initial begin
        if (HEAD_DIM < 1 || TILE_SIZE < 2 || MAX_CONTEXT < 1 || MAX_CONTEXT > 2047 ||
            HEAD_DIM % TILE_SIZE != 0 ||
            (TILE_SIZE & (TILE_SIZE-1)) != 0)
            $fatal(1, "Invalid single-column Attention configuration");
    end

    assign o_cmd_ready = (state == S_IDLE);
    assign o_kv_ready  = (state == S_WAIT_TILE);
    assign o_valid     = (state == S_RESPONSE);
    assign o_result    = result_q;

    always @* begin
        remaining_i = context_len_q - processed_keys_q;
        if (remaining_i <= 0)
            next_tile_count = 0;
        else if (remaining_i > TILE_SIZE)
            next_tile_count = TILE_SIZE;
        else
            next_tile_count = remaining_i[10:0];
    end

    // One TILE_SIZE x TILE_SIZE FP32 multiplier plane is shared by the three
    // phases. DOT maps PE=row(key) and lane=dimension; PV maps PE=dimension
    // and lane=key. Reduction trees use the corresponding orientation.
    wire [31:0] tile_product [0:TILE_SIZE-1][0:TILE_SIZE-1];
    wire [31:0] reduction_tree [0:TILE_SIZE-1][1:2*TILE_SIZE-1];
    wire [31:0] reduction_root [0:TILE_SIZE-1];

    wire [31:0] alpha_diff;
    wire [31:0] alpha_exp;
    wire [7:0] alpha_diff_status, alpha_exp_status;
    wire [31:0] beta_exp [0:TILE_SIZE-1];
    wire [31:0] beta_diff [0:TILE_SIZE-1];
    wire [7:0] beta_diff_status [0:TILE_SIZE-1];
    wire [7:0] beta_exp_status [0:TILE_SIZE-1];
    wire [31:0] online_max;
    wire [31:0] alpha_value = have_context_q ? alpha_exp : 32'b0;
    wire [31:0] shared_add_result [0:TILE_SIZE-1];
    wire [7:0] shared_add_status [0:TILE_SIZE-1];
    wire [31:0] final_inverse;
    wire [7:0] final_inverse_status;
    wire [TILE_SIZE-1:0] token_valid;

    reg [31:0] online_max_comb;
    reg [31:0] online_m_comb;
    integer max_i;
    always @* begin
        online_max_comb = 32'hff7fffff;
        for (max_i = 0; max_i < TILE_SIZE; max_i = max_i + 1)
            if (token_valid[max_i] && fp_gt(score_final[max_i], online_max_comb))
                online_max_comb = score_final[max_i];
        if (have_context_q)
            online_m_comb = fp_max(running_max_q, online_max_comb);
        else
            online_m_comb = online_max_comb;
    end
    assign online_max = online_m_comb;

    DW_fp_addsub #(23,8,1) u_alpha_diff (
        .a(running_max_q), .b(online_max), .rnd(3'b000), .op(1'b1),
        .z(alpha_diff), .status(alpha_diff_status));
    DW_fp_exp #(23,8,1,0) u_alpha_exp (
        .a(alpha_diff), .z(alpha_exp), .status(alpha_exp_status));

    genvar p, lane, tree_node;
    generate
        for (p = 0; p < TILE_SIZE; p = p + 1) begin : g_key_lane
            assign token_valid[p] = (processed_keys_q + p) < context_len_q;
            DW_fp_addsub #(23,8,1) u_beta_diff (
                .a(score_final[p]), .b(online_max), .rnd(3'b000), .op(1'b1),
                .z(beta_diff[p]), .status(beta_diff_status[p]));
            DW_fp_exp #(23,8,1,0) u_beta_exp (
                .a(beta_diff[p]), .z(beta_exp[p]), .status(beta_exp_status[p]));

            for (lane = 0; lane < TILE_SIZE; lane = lane + 1) begin : g_shared_mac
                wire [31:0] operand_a;
                wire [31:0] operand_b;
                wire [31:0] q_index = qk_chunk_q*TILE_SIZE + lane;
                wire [31:0] v_index = out_chunk_q*TILE_SIZE + lane;
                reg [31:0] mux_a, mux_b;
                always @* begin
                    mux_a = 32'b0;
                    mux_b = 32'b0;
                    case (state)
                        S_DOT: begin
                            mux_a = query_q[q_index*32 +: 32];
                            mux_b = key_q[(p*HEAD_DIM + q_index)*32 +: 32];
                        end
                        S_SCALE_QK: begin
                            if (p == 0) begin
                                mux_a = score_accum[lane];
                                mux_b = SCALE_BITS;
                            end
                        end
                        S_ONLINE: begin
                            if ((p == 0) && (lane == 0)) begin
                                mux_a = alpha_value;
                                mux_b = running_sum_q;
                            end
                        end
                        S_PV_SUM: begin
                            mux_a = beta_q[p];
                            mux_b = value_q[(p*HEAD_DIM + v_index)*32 +: 32];
                        end
                        S_PV_SCALE: begin
                            if (p == 0) begin
                                mux_a = alpha_q;
                                mux_b = output_accum[out_chunk_q*TILE_SIZE + lane];
                            end
                        end
                        S_FINAL_SCALE: begin
                            if (p == 0) begin
                                mux_a = output_accum[out_chunk_q*TILE_SIZE + lane];
                                mux_b = inverse_sum_q;
                            end
                        end
                        default: begin
                            mux_a = 32'b0;
                            mux_b = 32'b0;
                        end
                    endcase
                end
                assign operand_a = mux_a;
                assign operand_b = mux_b;
                DW_fp_mult #(23,8,1) u_tile_mac (
                    .a(operand_a), .b(operand_b), .rnd(3'b000),
                    .z(tile_product[p][lane]), .status());
            end
        end

        for (lane = 0; lane < TILE_SIZE; lane = lane + 1) begin : g_reduction
            for (tree_node = 1; tree_node < TILE_SIZE; tree_node = tree_node + 1) begin : g_add
                DW_fp_addsub #(23,8,1) u_tree_add (
                    .a(reduction_tree[lane][2*tree_node]),
                    .b(reduction_tree[lane][2*tree_node+1]),
                    .rnd(3'b000), .op(1'b0),
                    .z(reduction_tree[lane][tree_node]), .status());
            end
            assign reduction_root[lane] = reduction_tree[lane][1];
            for (tree_node = 0; tree_node < TILE_SIZE; tree_node = tree_node + 1) begin : g_leaf
                wire [31:0] dot_leaf = tile_product[lane][tree_node];
                wire [31:0] pv_leaf  = tile_product[tree_node][lane];
                wire [31:0] online_leaf = token_valid[tree_node] ? beta_exp[tree_node] : 32'b0;
                assign reduction_tree[lane][TILE_SIZE+tree_node] =
                    (state == S_DOT) ? dot_leaf :
                    (state == S_ONLINE && lane == 0) ? online_leaf :
                    ((state == S_PV_SUM || state == S_SCALE_QK ||
                      state == S_PV_SCALE || state == S_FINAL_SCALE) ? pv_leaf : 32'b0);
            end
        end
    endgenerate

    // QK partial accumulation, PV output accumulation, and online denominator
    // update share one TILE_SIZE-wide FP32 adder bank.
    generate
        for (lane = 0; lane < TILE_SIZE; lane = lane + 1) begin : g_shared_accum_add
            reg [31:0] add_a, add_b;
            always @* begin
                add_a = 32'b0;
                add_b = 32'b0;
                case (state)
                    S_DOT: begin
                        add_a = score_accum[lane];
                        add_b = reduction_root[lane];
                    end
                    S_ONLINE: begin
                        if (lane == 0) begin
                            add_a = tile_product[0][0];
                            add_b = reduction_root[0];
                        end
                    end
                    S_PV_SCALE: begin
                        add_a = pv_partial[out_chunk_q*TILE_SIZE + lane];
                        add_b = reduction_root[lane];
                    end
                    default: begin
                        add_a = 32'b0;
                        add_b = 32'b0;
                    end
                endcase
            end
            DW_fp_addsub #(23,8,1) u_shared_add (
                .a(add_a), .b(add_b), .rnd(3'b000), .op(1'b0),
                .z(shared_add_result[lane]), .status(shared_add_status[lane]));

        end
    endgenerate

    // The scalar divide is outside the tile. The final vector scaling reuses
    // the same 32x32 tile, one 32-element output slice per cycle.
    DW_fp_div #(23,8,1,0) u_final_inverse (
        .a(32'h3f800000), .b(running_sum_q), .rnd(3'b000),
        .z(final_inverse), .status(final_inverse_status));

    integer seq_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            context_len_q <= 0;
            processed_keys_q <= 0;
            tile_count_q <= 0;
            qk_chunk_q <= 0;
            out_chunk_q <= 0;
            running_max_q <= 0;
            running_sum_q <= 0;
            alpha_q <= 0;
            inverse_sum_q <= 0;
            have_context_q <= 0;
            result_q <= 0;
            o_error <= 0;
            o_cycle_count <= 0;
        end else begin
            if (state != S_IDLE && state != S_RESPONSE)
                o_cycle_count <= o_cycle_count + 1'b1;

            case (state)
                S_IDLE: begin
                    if (i_cmd_valid) begin
                        o_cycle_count <= 0;
                        o_error <= (i_context_len == 0 || i_context_len > MAX_CONTEXT);
                        if (i_context_len == 0 || i_context_len > MAX_CONTEXT) begin
                            result_q <= 0;
                            state <= S_RESPONSE;
                        end else begin
                            query_q <= i_query;
                            context_len_q <= i_context_len;
                            processed_keys_q <= 0;
                            running_max_q <= 0;
                            running_sum_q <= 0;
                            alpha_q <= 0;
                            have_context_q <= 0;
                            out_chunk_q <= 0;
                            for (seq_i = 0; seq_i < HEAD_DIM; seq_i = seq_i + 1)
                                output_accum[seq_i] <= 0;
                            state <= S_WAIT_TILE;
                        end
                    end
                end

                S_WAIT_TILE: begin
                    if (i_kv_valid) begin
                        key_q <= i_key_tile;
                        value_q <= i_value_tile;
                        tile_count_q <= next_tile_count;
                        qk_chunk_q <= 0;
                        for (seq_i = 0; seq_i < TILE_SIZE; seq_i = seq_i + 1)
                            score_accum[seq_i] <= 0;
                        state <= S_DOT;
                    end
                end

                S_DOT: begin
                    for (seq_i = 0; seq_i < TILE_SIZE; seq_i = seq_i + 1)
                        score_accum[seq_i] <= shared_add_result[seq_i];
                    if (qk_chunk_q == QK_CHUNKS-1) begin
                        state <= S_SCALE_QK;
                    end else begin
                        qk_chunk_q <= qk_chunk_q + 1'b1;
                    end
                end

                S_SCALE_QK: begin
                    for (seq_i = 0; seq_i < TILE_SIZE; seq_i = seq_i + 1)
                        score_final[seq_i] <= reduction_root[seq_i];
                    state <= S_ONLINE;
                end

                S_ONLINE: begin
                    running_max_q <= online_max;
                    running_sum_q <= shared_add_result[0];
                    alpha_q <= alpha_value;
                    have_context_q <= 1'b1;
                    for (seq_i = 0; seq_i < TILE_SIZE; seq_i = seq_i + 1)
                        beta_q[seq_i] <= token_valid[seq_i] ? beta_exp[seq_i] : 32'b0;
                    out_chunk_q <= 0;
                    state <= S_PV_SUM;
                end

                S_PV_SUM: begin
                    for (seq_i = 0; seq_i < TILE_SIZE; seq_i = seq_i + 1)
                        pv_partial[out_chunk_q*TILE_SIZE + seq_i] <= reduction_root[seq_i];
                    state <= S_PV_SCALE;
                end

                S_PV_SCALE: begin
                    for (seq_i = 0; seq_i < TILE_SIZE; seq_i = seq_i + 1)
                        output_accum[out_chunk_q*TILE_SIZE + seq_i] <= shared_add_result[seq_i];
                    if (out_chunk_q == OUT_CHUNKS-1) begin
                        processed_keys_q <= processed_keys_q + tile_count_q;
                        if (processed_keys_q + tile_count_q >= context_len_q)
                            state <= S_FINAL_INV;
                        else
                            state <= S_WAIT_TILE;
                    end else begin
                        out_chunk_q <= out_chunk_q + 1'b1;
                        state <= S_PV_SUM;
                    end
                end

                S_FINAL_INV: begin
                    inverse_sum_q <= final_inverse;
                    out_chunk_q <= 0;
                    state <= S_FINAL_SCALE;
                end

                S_FINAL_SCALE: begin
                    for (seq_i = 0; seq_i < TILE_SIZE; seq_i = seq_i + 1)
                        result_q[(out_chunk_q*TILE_SIZE + seq_i)*32 +: 32] <= reduction_root[seq_i];
                    if (out_chunk_q == OUT_CHUNKS-1)
                        state <= S_RESPONSE;
                    else
                        out_chunk_q <= out_chunk_q + 1'b1;
                end

                S_RESPONSE: begin
                    if (i_ready)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
