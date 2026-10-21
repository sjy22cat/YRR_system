`timescale 1ns/1ps
// Iterative balanced maximum reduction. One compare level is evaluated per
// cycle, with at most LANES comparators active, to avoid a deep 1024-way path.
// The caller must hold in_vector/in_len stable from start through out_valid;
// the top-level Softmax wrapper supplies its request registers directly.
module softmax_parallel_max #(
 parameter VEC_LEN=1024,
 parameter LANES=256
)(
 input wire clk,rst_n,
 input wire start,
 output wire ready,
 input wire [VEC_LEN*32-1:0] in_vector,
 input wire [10:0] in_len,
 output reg out_valid,
 output reg [31:0] max_value,
 output reg input_error
);
 localparam PAD_LEN=2**$clog2(VEC_LEN);
 localparam MAX_NODES=(PAD_LEN/2>0)?PAD_LEN/2:1;
 localparam REDUCE_LANES=(LANES>MAX_NODES)?MAX_NODES:LANES;
 localparam FIRST_BATCHES=(MAX_NODES+REDUCE_LANES-1)/REDUCE_LANES;
 localparam TREE_LEVELS=$clog2(PAD_LEN);
 localparam IDLE=0,REDUCE=1;

 reg state;
 reg [10:0] len_q,base_idx;
 reg [4:0] level_q;
 reg error_q;
 reg [31:0] work_q[0:MAX_NODES-1];
 wire [31:0] pair_value[0:REDUCE_LANES-1];
 wire [REDUCE_LANES-1:0] pair_error;
 wire [10:0] node_count=PAD_LEN>>(level_q+1'b1);

 function automatic [31:0] fp_max(input [31:0] a,b);
  begin
   if(a[31]!=b[31])fp_max=a[31]?b:a;
   else if(a[31])fp_max=a<b?a:b;
   else fp_max=a>b?a:b;
  end
 endfunction
 function automatic finite(input [31:0] a);
  finite=a[30:23]!=8'hff;
 endfunction

 initial begin
  if(VEC_LEN<1||VEC_LEN>1024||LANES<1||
    (LANES&(LANES-1))!=0)
   $fatal(1,"Invalid maximum-reduction lane configuration");
 end

 assign ready=state==IDLE;
 genvar g;
 generate
  for(g=0;g<REDUCE_LANES;g=g+1)begin:g_compare
   wire [10:0] pair_index=base_idx+g;
   wire [10:0] left_index=pair_index*2;
   wire [10:0] right_index=left_index+1'b1;
   wire active=pair_index<node_count;
   wire [31:0] left_first[FIRST_BATCHES-1:0];
   wire [31:0] right_first[FIRST_BATCHES-1:0];
   reg [31:0] first_left_selected,first_right_selected;
   for(genvar b=0;b<FIRST_BATCHES;b=b+1)begin:g_first_batch
    localparam LEFT_INDEX=2*(b*REDUCE_LANES+g);
    localparam RIGHT_INDEX=LEFT_INDEX+1;
    if(LEFT_INDEX<VEC_LEN)
     assign left_first[b]=in_vector[LEFT_INDEX*32+:32];
    else assign left_first[b]=32'hff7fffff;
    if(RIGHT_INDEX<VEC_LEN)
     assign right_first[b]=in_vector[RIGHT_INDEX*32+:32];
    else assign right_first[b]=32'hff7fffff;
   end
   always @*begin
    first_left_selected=32'hff7fffff;
    first_right_selected=32'hff7fffff;
    for(integer b=0;b<FIRST_BATCHES;b=b+1)
     if(base_idx==b*REDUCE_LANES)begin
      first_left_selected=left_first[b];
      first_right_selected=right_first[b];
     end
   end
   wire [31:0] tree_left,tree_right;
   if(2*g+1<MAX_NODES)begin:g_tree_child_valid
    assign tree_left=work_q[2*g];
    assign tree_right=work_q[2*g+1];
   end else begin:g_tree_child_pad
    assign tree_left=32'hff7fffff;
    assign tree_right=32'hff7fffff;
   end
   wire [31:0] left_value=(level_q==0)?
    ((left_index<len_q)?first_left_selected:32'hff7fffff):
    ((g<node_count)?tree_left:32'hff7fffff);
   wire [31:0] right_value=(level_q==0)?
    ((right_index<len_q)?first_right_selected:32'hff7fffff):
    ((g<node_count)?tree_right:32'hff7fffff);
   assign pair_value[g]=fp_max(left_value,right_value);
   assign pair_error[g]=(level_q==0&&active&&
    ((left_index<len_q&&!finite(left_value))||
     (right_index<len_q&&!finite(right_value))));
  end
 endgenerate

 always @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin
   state<=IDLE;len_q<=0;base_idx<=0;level_q<=0;error_q<=0;
   out_valid<=0;max_value<=0;input_error<=0;
  end else begin
   out_valid<=0;
   case(state)
    IDLE:if(start)begin
     len_q<=in_len;base_idx<=0;level_q<=0;error_q<=0;input_error<=0;
     if(PAD_LEN==1)begin
      max_value<=in_vector[31:0];
      input_error<=in_len!=0&&!finite(in_vector[31:0]);
      out_valid<=1;
     end else state<=REDUCE;
    end
    REDUCE:begin
     for(integer j=0;j<REDUCE_LANES;j=j+1)
      if(base_idx+j<node_count)begin
       if(level_q==TREE_LEVELS-1)begin
        max_value<=pair_value[j];
        input_error<=error_q||(|pair_error);
        out_valid<=1;
        state<=IDLE;
       end else work_q[base_idx+j]<=pair_value[j];
      end
     if(|pair_error)error_q<=1;
     if(level_q!=TREE_LEVELS-1&&base_idx+REDUCE_LANES>=node_count)begin
      base_idx<=0;level_q<=level_q+1'b1;
     end else if(level_q!=TREE_LEVELS-1)
      base_idx<=base_idx+REDUCE_LANES;
    end
    default:state<=IDLE;
   endcase
  end
 end
endmodule
