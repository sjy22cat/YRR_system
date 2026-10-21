`timescale 1ns/1ps
// Standalone Softmax with a cycle-by-cycle maximum reduction followed by
// time-shared, staged elementwise and sum-reduction engines.
module softmax_1024_parallel #(
 parameter VEC_LEN=1024,
 parameter LANES=256,
 parameter EXP_LANES=128
)(
 input wire clk,rst_n,
 input wire cmd_valid,
 output wire cmd_ready,
 input wire [10:0] cmd_len,
 input wire [31:0] cmd_tag,
 input wire [VEC_LEN*32-1:0] cmd_vector,
 output wire out_valid,
 input wire out_ready,
 output wire [VEC_LEN*32-1:0] out_vector,
 output reg [10:0] out_len,
 output reg [31:0] out_tag,
 output reg [31:0] out_aux,
 output wire out_error,
 output wire busy,
 output reg [31:0] cycle_count
);
 localparam IDLE=0,WAIT_MAX=1,WAIT_CALC=2,ERROR_RESPONSE=3;
 reg [1:0] state;
 reg [VEC_LEN*32-1:0] input_q;
 reg [10:0] len_q;
 reg [31:0] tag_q;

 wire max_ready,max_valid,max_error;
 wire [31:0] max_value;
 wire calc_ready,calc_valid,calc_error;
 wire [VEC_LEN*32-1:0] calc_vector;
 wire [10:0] calc_len;
 wire [31:0] calc_tag,calc_max,calc_cycles;
 wire max_start,calc_start,calc_out_ready;

 assign cmd_ready=state==IDLE&&max_ready&&calc_ready;
 assign busy=state!=IDLE;
 // Start reduction one cycle after acceptance, when input_q already contains
 // the stable request vector shared by the maximum and calculation stages.
 assign max_start=state==WAIT_MAX&&max_ready&&!max_valid;
 assign calc_start=state==WAIT_MAX&&max_valid&&!max_error;
 assign calc_out_ready=state==WAIT_CALC&&out_ready;
 assign out_valid=state==ERROR_RESPONSE||
  (state==WAIT_CALC&&calc_valid);
 assign out_error=state==ERROR_RESPONSE||
  (state==WAIT_CALC&&calc_error);
 assign out_vector=(state==WAIT_CALC)?calc_vector:'0;

 softmax_parallel_max #(.VEC_LEN(VEC_LEN),.LANES(LANES)) u_max(
  .clk(clk),.rst_n(rst_n),.start(max_start),.ready(max_ready),
  .in_vector(input_q),.in_len(len_q),.out_valid(max_valid),
  .max_value(max_value),.input_error(max_error));

 softmax_parallel_calc #(
  .VEC_LEN(VEC_LEN),.LANES(LANES),.EXP_LANES(EXP_LANES)) u_calc(
  .clk(clk),.rst_n(rst_n),.start(calc_start),.ready(calc_ready),
  .cmd_len(len_q),.cmd_tag(tag_q),.max_value(max_value),.in_vector(input_q),
  .out_valid(calc_valid),.out_ready(calc_out_ready),.out_vector(calc_vector),
  .out_len(calc_len),.out_tag(calc_tag),.out_max(calc_max),
  .out_error(calc_error),.cycle_count(calc_cycles));

 always @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin
   state<=IDLE;len_q<=0;tag_q<=0;
   out_len<=0;out_tag<=0;out_aux<=0;cycle_count<=0;
  end else begin
   if(state!=IDLE&&!out_valid)cycle_count<=cycle_count+1;
   case(state)
    IDLE:if(cmd_valid&&cmd_ready)begin
     input_q<=cmd_vector;len_q<=cmd_len;tag_q<=cmd_tag;
     out_len<=cmd_len;out_tag<=cmd_tag;out_aux<=0;cycle_count<=0;
     if(cmd_len==0||cmd_len>VEC_LEN)state<=ERROR_RESPONSE;
     else state<=WAIT_MAX;
    end
    WAIT_MAX:if(max_valid)begin
     out_aux<=max_value;
     if(max_error)state<=ERROR_RESPONSE;
     else state<=WAIT_CALC;
    end
    WAIT_CALC:if(calc_valid&&out_ready)state<=IDLE;
    ERROR_RESPONSE:if(out_ready)state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule
