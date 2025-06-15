#include "pipeline-lstm-header.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "ap_axi_sdata.h"
#include "hls_stream.h"

using namespace std;

int main(){

//Output/Input definitions
	hls::stream<ap_axis<32,2,5,6>> final_output;
	hls::stream<ap_axis<32,2,5,6>> x_input_final;

	ifstream inp_file;
	inp_file.open("/home/temporary/Desktop/QLSTM_HLS/qlstm_finn_model_fpl_24/hlslib_solution/hls_test_x_int5.txt");
	//Reading test inputs in the below loop line by line and storing it in an input test array.
	string inp_line0,inp_val0;
	int inp_row_mm=0;
		 while (getline(inp_file, inp_line0) && inp_row_mm < num_test_inputs) {
			 	 istringstream inp_iss0(inp_line0);
			 	 int inp_col = 0;
		         while (getline(inp_iss0, inp_val0, ',')  && inp_col < num_test_inputs*20) { //Within a line, reading values one by one with ',' as delimiter.
		        	 ap_axis<32,2,5,6> data_inp;
		        	 data_inp.data = stoi(inp_val0);
		        	 x_input_final.write(data_inp);
		        	 inp_col++;
		         }
		        inp_row_mm++;
		 }

	//Module instantiation for testing	
	qlstm_top_2(x_input_final,final_output);

	for(int i=0;i<num_test_inputs*Out_N;++i){
	ap_axis<32,2,5,6> data_out;
	final_output.read(data_out);
	cout << data_out.data << "   " << i << std::endl;
	}
	return 0;
}
