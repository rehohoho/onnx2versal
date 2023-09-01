#ifndef IDENTITY_H_
#define IDENTITY_H_

#include <adf.h>

/** 
 * @defgroup IdentityKernels
 * @ingroup Identity
 * 
 * @{
 */

/**
 * @brief Scalar implementation,
 * Identity<8> total = 35
 */
template <typename TT, int N>
class Identity {
  public:
    void filter(
      input_stream<TT>* in,
      output_stream<TT>* out
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(Identity::filter);
    }
};
/** @}*/


#endif // IDENTITY_H_
