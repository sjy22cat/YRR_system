`timescale 1ns/1ps
// Post-maximum Softmax engine. Elementwise operators are time-shared across
// vector chunks. Subtract, exp, each sum-tree level, divide, and multiply are
// separate cycles so no cycle contains a cascade of floating-point operators.
// Operand staging registers keep vector-bank selection off the FP critical
// paths. The extra preparation states are paid once per request, while the
// elementwise engines retain one result batch per cycle.
module softmax_parallel_calc #(
 parameter VEC_LEN=1024,
 parameter LANES=256,
 parameter EXP_LANES=128
)(
 input wire clk,rst_n,
 input wire start,
 output wire ready,
 input wire [10:0] cmd_len,
 input wire [31:0] cmd_tag,
 input wire [31:0] max_value,
 input wire [VEC_LEN*32-1:0] in_vector,
 output wire out_valid,
 input wire out_ready,
 output wire [VEC_LEN*32-1:0] out_vector,
 output wire [10:0] out_len,
 output wire [31:0] out_tag,out_max,
 output wire out_error,
 output reg [31:0] cycle_count
);
 localparam PAD_LEN=2**$clog2(VEC_LEN);
 localparam SUM_NODES=(PAD_LEN/2>0)?PAD_LEN/2:1;
 localparam ELEMENT_COUNT=(LANES>PAD_LEN)?PAD_LEN:LANES;
 localparam EXP_COUNT=(EXP_LANES>PAD_LEN)?PAD_LEN:EXP_LANES;
 localparam SUM_COUNT=(LANES>SUM_NODES)?SUM_NODES:LANES;
 localparam ELEMENT_BATCHES=(PAD_LEN+ELEMENT_COUNT-1)/ELEMENT_COUNT;
 localparam EXP_BATCHES=(PAD_LEN+EXP_COUNT-1)/EXP_COUNT;
 localparam SUM_BATCHES=(SUM_NODES+SUM_COUNT-1)/SUM_COUNT;
 localparam TREE_LEVELS=$clog2(PAD_LEN);
 localparam IDLE=0,SUBTRACT=1,EXP_PREP=2,EXPONENT=3,SUM_PREP=4,
  SUM_REDUCE=5,INVERSE=6,MULTIPLY=7,RESPONSE=8;

 reg [3:0] state;
 reg [10:0] len_q,base_idx,sum_base;
 reg [4:0] sum_level;
 reg [31:0] tag_q,max_q,stat_q,inverse_q;
 reg [VEC_LEN*32-1:0] vector_q;
 reg error_q,sum_error_q;
 reg [31:0] sum_work[0:SUM_NODES-1];
 reg [31:0] element_input_q[0:ELEMENT_COUNT-1];
 reg [31:0] exp_input_q[0:EXP_COUNT-1];
 reg [31:0] sum_first_left_q[0:SUM_COUNT-1];
 reg [31:0] sum_first_right_q[0:SUM_COUNT-1];

 wire [ELEMENT_COUNT*32-1:0] element_next_inputs;
 wire [EXP_COUNT*32-1:0] exp_first_inputs,exp_next_inputs;
 wire [SUM_COUNT*32-1:0] sum_first_left_inputs,sum_first_right_inputs;
 wire [SUM_COUNT*32-1:0] sum_next_left_inputs,sum_next_right_inputs;
 wire [ELEMENT_COUNT*32-1:0] sub_values,mul_values;
 wire [ELEMENT_COUNT-1:0] sub_error,mul_error;
 wire [EXP_COUNT*32-1:0] exp_values;
 wire [EXP_COUNT-1:0] exp_error;
 wire [SUM_COUNT*32-1:0] sum_left_inputs,sum_right_inputs;
 wire [SUM_COUNT*32-1:0] sum_values;
 wire [SUM_COUNT-1:0] sum_lane_error;
 wire [10:0] sum_node_count=PAD_LEN>>(sum_level+1'b1);
 wire [31:0] inverse;
 wire [7:0] inverse_status;

 function automatic finite(input [31:0] a);
  finite=a[30:23]!=8'hff;
 endfunction

 initial begin
  if(VEC_LEN<1||VEC_LEN>1024||LANES<1||
    (LANES&(LANES-1))!=0||EXP_LANES<1||
    (EXP_LANES&(EXP_LANES-1))!=0)
   $fatal(1,"Invalid parallel Softmax lane configuration");
 end

 assign ready=state==IDLE;
 assign out_valid=state==RESPONSE;
 assign out_len=len_q;
 assign out_tag=tag_q;
 assign out_max=max_q;
 assign out_error=error_q;
 genvar g;
 generate
  for(g=0;g<VEC_LEN;g=g+1)begin:g_output
   assign out_vector[g*32+:32]=
    (!error_q&&state==RESPONSE&&g<len_q)?vector_q[g*32+:32]:32'b0;
  end
  for(g=0;g<ELEMENT_COUNT;g=g+1)begin:g_element_lane
   wire [10:0] idx=base_idx+g;
   wire active=idx<len_q&&idx<VEC_LEN;
   wire [31:0] input_words[ELEMENT_BATCHES-1:0];
   reg [31:0] next_input_selected;
   for(genvar b=0;b<ELEMENT_BATCHES;b=b+1)begin:g_element_batch
    localparam WORD_INDEX=b*ELEMENT_COUNT+g;
    if(WORD_INDEX<VEC_LEN)
     assign input_words[b]=vector_q[WORD_INDEX*32+:32];
    else assign input_words[b]=32'b0;
   end
   always @*begin
    next_input_selected=32'b0;
    for(integer b=0;b<ELEMENT_BATCHES;b=b+1)
     if(base_idx+ELEMENT_COUNT==b*ELEMENT_COUNT)
      next_input_selected=input_words[b];
   end
   assign element_next_inputs[g*32+:32]=next_input_selected;
   wire [31:0] add_value,mul_value;
   wire [31:0] add_a,add_b;
   wire [7:0] add_status,mul_status;
   wire add_sub;
   if(g<SUM_COUNT)begin:g_shared_add_operands
    assign add_a=state==SUBTRACT?(active?element_input_q[g]:32'b0):
     state==SUM_REDUCE?sum_left_inputs[g*32+:32]:32'b0;
    assign add_b=state==SUBTRACT?(active?max_q:32'b0):
     state==SUM_REDUCE?sum_right_inputs[g*32+:32]:32'b0;
    assign sum_values[g*32+:32]=add_value;
   end else begin:g_sub_only_operands
    assign add_a=state==SUBTRACT&&active?element_input_q[g]:32'b0;
    assign add_b=state==SUBTRACT&&active?max_q:32'b0;
   end
   assign add_sub=state==SUBTRACT;
   DW_fp_addsub #(23,8,1) u_addsub(
    add_a,add_b,3'b000,add_sub,add_value,add_status);
   DW_fp_mult #(23,8,1) u_mul(
    state==MULTIPLY&&active?element_input_q[g]:32'b0,
    state==MULTIPLY&&active?inverse_q:32'b0,
    3'b000,mul_value,mul_status);
   assign sub_values[g*32+:32]=add_value;
   assign mul_values[g*32+:32]=mul_value;
   assign sub_error[g]=state==SUBTRACT&&active&&!finite(add_value);
   assign mul_error[g]=state==MULTIPLY&&active&&!finite(mul_value);
  end
  for(g=0;g<EXP_COUNT;g=g+1)begin:g_exp_lane
   wire [10:0] idx=base_idx+g;
   wire active=idx<len_q&&idx<VEC_LEN;
   wire [31:0] input_words[EXP_BATCHES-1:0];
   reg [31:0] next_input_selected;
   for(genvar b=0;b<EXP_BATCHES;b=b+1)begin:g_exp_batch
    localparam WORD_INDEX=b*EXP_COUNT+g;
    if(WORD_INDEX<VEC_LEN)
     assign input_words[b]=vector_q[WORD_INDEX*32+:32];
    else assign input_words[b]=32'b0;
   end
   always @*begin
    next_input_selected=32'b0;
    for(integer b=0;b<EXP_BATCHES;b=b+1)
     if(base_idx+EXP_COUNT==b*EXP_COUNT)
      next_input_selected=input_words[b];
   end
   assign exp_first_inputs[g*32+:32]=input_words[0];
   assign exp_next_inputs[g*32+:32]=next_input_selected;
   wire [31:0] exp_value;
   wire [7:0] exp_status;
   DW_fp_exp #(23,8,1,0) u_exp(
    state==EXPONENT&&active?exp_input_q[g]:32'b0,exp_value,exp_status);
   assign exp_values[g*32+:32]=exp_value;
   assign exp_error[g]=state==EXPONENT&&active&&!finite(exp_value);
  end
  for(g=0;g<SUM_COUNT;g=g+1)begin:g_sum_lane
   wire [10:0] idx=sum_base+g;
   wire active=idx<sum_node_count;
   wire [31:0] left_words[SUM_BATCHES-1:0];
   wire [31:0] right_words[SUM_BATCHES-1:0];
   reg [31:0] next_left_selected,next_right_selected;
   for(genvar b=0;b<SUM_BATCHES;b=b+1)begin:g_sum_batch
    localparam LEFT_INDEX=2*(b*SUM_COUNT+g);
    localparam RIGHT_INDEX=LEFT_INDEX+1;
    if(LEFT_INDEX<VEC_LEN)
     assign left_words[b]=vector_q[LEFT_INDEX*32+:32];
    else assign left_words[b]=32'b0;
    if(RIGHT_INDEX<VEC_LEN)
     assign right_words[b]=vector_q[RIGHT_INDEX*32+:32];
    else assign right_words[b]=32'b0;
   end
   always @*begin
    next_left_selected=32'b0;
    next_right_selected=32'b0;
    for(integer b=0;b<SUM_BATCHES;b=b+1)
     if(sum_base+SUM_COUNT==b*SUM_COUNT)begin
      next_left_selected=left_words[b];
      next_right_selected=right_words[b];
     end
   end
   assign sum_first_left_inputs[g*32+:32]=left_words[0];
   assign sum_first_right_inputs[g*32+:32]=right_words[0];
   assign sum_next_left_inputs[g*32+:32]=next_left_selected;
   assign sum_next_right_inputs[g*32+:32]=next_right_selected;
   wire [31:0] tree_left,tree_right;
   if(2*g+1<SUM_NODES)begin:g_sum_child_valid
    assign tree_left=sum_work[2*g];
    assign tree_right=sum_work[2*g+1];
   end else begin:g_sum_child_pad
    assign tree_left=32'b0;
    assign tree_right=32'b0;
   end
   wire [31:0] left_value=(sum_level==0)?
    sum_first_left_q[g]:
    ((g<sum_node_count)?tree_left:32'b0);
   wire [31:0] right_value=(sum_level==0)?
    sum_first_right_q[g]:
    ((g<sum_node_count)?tree_right:32'b0);
   assign sum_left_inputs[g*32+:32]=active?left_value:32'b0;
   assign sum_right_inputs[g*32+:32]=active?right_value:32'b0;
   assign sum_lane_error[g]=active&&!finite(sum_values[g*32+:32]);
  end
 endgenerate

 DW_fp_div #(23,8,1,0) u_inverse(
  32'h3f800000,stat_q,3'b000,inverse,inverse_status);

 always @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin
   state<=IDLE;len_q<=0;base_idx<=0;sum_base<=0;sum_level<=0;
   tag_q<=0;max_q<=0;stat_q<=0;inverse_q<=0;
   error_q<=0;sum_error_q<=0;cycle_count<=0;
   for(integer r=0;r<ELEMENT_COUNT;r=r+1)begin
    element_input_q[r]<=0;
   end
   for(integer r=0;r<EXP_COUNT;r=r+1)
    exp_input_q[r]<=0;
   for(integer r=0;r<SUM_COUNT;r=r+1)begin
    sum_first_left_q[r]<=0;
    sum_first_right_q[r]<=0;
   end
  end else begin
   if(state!=IDLE&&state!=RESPONSE)cycle_count<=cycle_count+1;
   case(state)
    IDLE:if(start)begin
     vector_q<=in_vector;len_q<=cmd_len;tag_q<=cmd_tag;max_q<=max_value;
     base_idx<=0;sum_base<=0;sum_level<=0;stat_q<=0;inverse_q<=0;
     error_q<=cmd_len==0||cmd_len>VEC_LEN;sum_error_q<=0;cycle_count<=0;
     for(integer w=0;w<ELEMENT_COUNT;w=w+1)
      element_input_q[w]<=w<VEC_LEN?in_vector[w*32+:32]:32'b0;
     state<=(cmd_len==0||cmd_len>VEC_LEN)?RESPONSE:SUBTRACT;
    end
    SUBTRACT:begin
     for(integer w=0;w<ELEMENT_COUNT;w=w+1)
      if(base_idx+w<len_q&&base_idx+w<VEC_LEN)
       vector_q[(base_idx+w)*32+:32]<=sub_values[w*32+:32];
     if(base_idx+ELEMENT_COUNT<len_q)
      for(integer w=0;w<ELEMENT_COUNT;w=w+1)
       element_input_q[w]<=element_next_inputs[w*32+:32];
     if(|sub_error)error_q<=1;
     if(base_idx+ELEMENT_COUNT>=len_q)begin
      base_idx<=0;state<=EXP_PREP;
     end else base_idx<=base_idx+ELEMENT_COUNT;
    end
    EXP_PREP:begin
     for(integer w=0;w<EXP_COUNT;w=w+1)
      exp_input_q[w]<=w<len_q?exp_first_inputs[w*32+:32]:32'b0;
     base_idx<=0;state<=EXPONENT;
    end
    EXPONENT:begin
     for(integer w=0;w<EXP_COUNT;w=w+1)
      if(base_idx+w<len_q&&base_idx+w<VEC_LEN)
       vector_q[(base_idx+w)*32+:32]<=exp_values[w*32+:32];
     if(base_idx+EXP_COUNT<len_q)
      for(integer w=0;w<EXP_COUNT;w=w+1)
       exp_input_q[w]<=exp_next_inputs[w*32+:32];
     if(|exp_error)error_q<=1;
     if(base_idx+EXP_COUNT>=len_q)begin
      base_idx<=0;sum_base<=0;sum_level<=0;sum_error_q<=0;
      state<=SUM_PREP;
     end else base_idx<=base_idx+EXP_COUNT;
    end
    SUM_PREP:begin
     for(integer w=0;w<SUM_COUNT;w=w+1)begin
      sum_first_left_q[w]<=2*w<len_q?
       sum_first_left_inputs[w*32+:32]:32'b0;
      sum_first_right_q[w]<=2*w+1<len_q?
       sum_first_right_inputs[w*32+:32]:32'b0;
     end
     sum_base<=0;sum_level<=0;sum_error_q<=0;state<=SUM_REDUCE;
    end
    SUM_REDUCE:begin
     if(PAD_LEN==1)begin
      stat_q<=vector_q[31:0];
      if(!finite(vector_q[31:0]))error_q<=1;
      state<=INVERSE;
     end else begin
      for(integer w=0;w<SUM_COUNT;w=w+1)
       if(sum_base+w<sum_node_count)begin
        if(sum_level==TREE_LEVELS-1)begin
         stat_q<=sum_values[w*32+:32];
         if(sum_error_q||(|sum_lane_error)||!finite(sum_values[w*32+:32]))
          error_q<=1;
         state<=INVERSE;
        end else sum_work[sum_base+w]<=sum_values[w*32+:32];
       end
      if(|sum_lane_error)sum_error_q<=1;
      if(sum_level!=TREE_LEVELS-1&&sum_base+SUM_COUNT>=sum_node_count)begin
       sum_base<=0;sum_level<=sum_level+1'b1;
      end else if(sum_level!=TREE_LEVELS-1)
       begin
       if(sum_level==0)
        for(integer w=0;w<SUM_COUNT;w=w+1)begin
         sum_first_left_q[w]<=2*(sum_base+SUM_COUNT+w)<len_q?
          sum_next_left_inputs[w*32+:32]:32'b0;
         sum_first_right_q[w]<=2*(sum_base+SUM_COUNT+w)+1<len_q?
          sum_next_right_inputs[w*32+:32]:32'b0;
        end
       sum_base<=sum_base+SUM_COUNT;
       end
     end
    end
    INVERSE:begin
     inverse_q<=inverse;
     for(integer w=0;w<ELEMENT_COUNT;w=w+1)
      element_input_q[w]<=w<len_q&&w<VEC_LEN?
       vector_q[w*32+:32]:32'b0;
     if(!finite(inverse))error_q<=1;
     base_idx<=0;state<=MULTIPLY;
    end
    MULTIPLY:begin
     for(integer w=0;w<ELEMENT_COUNT;w=w+1)
      if(base_idx+w<len_q&&base_idx+w<VEC_LEN)
       vector_q[(base_idx+w)*32+:32]<=mul_values[w*32+:32];
     if(base_idx+ELEMENT_COUNT<len_q)
      for(integer w=0;w<ELEMENT_COUNT;w=w+1)
       element_input_q[w]<=element_next_inputs[w*32+:32];
     if(|mul_error)error_q<=1;
     if(base_idx+ELEMENT_COUNT>=len_q)begin
      base_idx<=0;state<=RESPONSE;
     end else base_idx<=base_idx+ELEMENT_COUNT;
    end
    RESPONSE:if(out_ready)state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule
