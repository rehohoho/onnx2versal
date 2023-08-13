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
 * TransposeScalarBHWC2BCHW::filter<f,1,4,4,16>, total = 371
 */
template <typename TT, int B, int H, int W, int C, int PAD_W>
class TransposeScalarBHWC2BCHW {
  public:
    void filter(
      input_window<TT>* in,
      output_window<TT>* out
    );

    static void registerKernelClass() {
      static_assert(W == PAD_W);
      REGISTER_FUNCTION(TransposeScalarBHWC2BCHW::filter);
    }
};


/**
 * @brief Scalar implementation for BHWC to BCHW,
 * TransposeScalarBCHW2BHWC::filter<f,1,4,4,16>, total = 334
 */
template <typename TT, int B, int H, int W, int C, int PAD_W>
class TransposeScalarBCHW2BHWC {
  public:
    void filter(
      input_window<TT>* in,
      output_window<TT>* out
    );

    static void registerKernelClass() {
      REGISTER_FUNCTION(TransposeScalarBCHW2BHWC::filter);
    }
};


/**
 * @brief Scalar stream implementation for BHWC to BCHW,
 * TransposeScalarBHWC2BCHW::filter<f,1,4,4,16>, total = 330
 */
template <typename TT, int B, int H, int W, int C, int PAD_W>
class TransposeScalarBHWC2BCHWStream {
  public:
    void filter(
      input_window<TT>* in,
      output_stream<TT>* restrict out
    );

    static void registerKernelClass() {
      static_assert(sizeof(TT) == 4);
      REGISTER_FUNCTION(TransposeScalarBHWC2BCHWStream::filter);
    }
};

/**
 * @brief Scalar implementation using input pktstream for BHWC to BCHW,
 */
template <typename TT, int B, int H, int W, int C, int PAD_W>
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
