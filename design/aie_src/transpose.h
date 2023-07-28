#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include <adf.h>


/** 
 * @defgroup TransposeKernels
 * @ingroup Transpose
 * 
 * @{
 */

/**
 * @brief Scalar implementation for BHWC to BCHW,
 * TransposeScalarBHWC2BCHW::filter<f,1,4,4,16>, total = 330
 */
template <typename TT, int B, int H, int W, int C>
class TransposeScalarBHWC2BCHW {
  public:
    void filter(
      input_window<TT>* in,
      output_stream<TT>* restrict out
    );

    static void registerKernelClass() {
      REGISTER_FUNCTION(TransposeScalarBHWC2BCHW::filter);
    }
};


/**
 * @brief Scalar implementation using input pktstream for BHWC to BCHW,
 */
template <typename TT, int B, int H, int W, int C>
class TransposeScalarPktStreamBHWC2BCHW {
  public:
    void filter(
      input_pktstream* in,
      output_window<TT>* out
    );

    static void registerKernelClass() {
      REGISTER_FUNCTION(TransposeScalarPktStreamBHWC2BCHW::filter);
    }
};
/** @}*/


#endif // TRANSPOSE_H_
