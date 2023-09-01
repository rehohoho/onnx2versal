#include "identity.h"
#include "kernel_utils.h"


template <typename TT, int N>
void Identity<TT, N>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER(printf("Running Identity<%d>\n", N));

  static constexpr int NSTEP = 16 / sizeof(TT);

  for (int n = 0; n < N; n += NSTEP)
    put_wms(0, get_wss(0));

  PROFILE_FOOTER;
}
