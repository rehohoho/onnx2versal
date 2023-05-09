#include "identity.h"
#include "kernel_utils.h"


template <int N>
void IdentityScalar<N>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf("Running IdentityScalar<%d>\n", N));

  for (int i = 0; i < N; i++)
    window_writeincr(out, window_readincr(in));

  PROFILE_FOOTER;
}
