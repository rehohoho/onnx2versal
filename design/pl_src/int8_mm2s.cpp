#include <iostream>
#include <hls_stream.h>

typedef int8_t dinp_t;
typedef int dout_t;

extern "C" {


// int8 or char stream has compile issue for sw_emu, pl-aie axis has minimum of 32-bits?
void int8_mm2s(dinp_t* mem, hls::stream<dout_t>& s, int size) {
#pragma HLS INTERFACE m_axi port=mem offset=slave bundle=gmem
#pragma HLS INTERFACE axis port=s
#pragma HLS INTERFACE s_axilite port=mem bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS interface s_axilite port=return bundle=control

	for(int i = 0; i < size; i+=4) {
#pragma HLS PIPELINE II=1
		s.write(mem[i+0]);
		s.write(mem[i+1]);
		s.write(mem[i+2]);
		s.write(mem[i+3]);
	}
}


}
