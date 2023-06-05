#include <iostream>
#include <hls_stream.h>

typedef float dinp_t;
typedef float dout_t;

extern "C" {


void float32_s2mm(dinp_t* mem, hls::stream<dout_t>& s, int size) {
#pragma HLS INTERFACE m_axi port=mem offset=slave bundle=gmem
#pragma HLS INTERFACE axis port=s
#pragma HLS INTERFACE s_axilite port=mem bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS interface s_axilite port=return bundle=control

	for(int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
		dout_t x = s.read();
		mem[i] = x;
		std::cout << x << " " << std::endl;
	}
}


}
