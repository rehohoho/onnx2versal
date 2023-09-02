#ifndef POOL_H_
#define POOL_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup Pool2DKernels
 * @ingroup Pool2D
 * 
 * @details
 * Design Notes
 * - Bandwidth limited
 * - 2 accs causes spilling
 * - All kernels assume INP_W divisible by OUT_W
 * 
 * @{
 */

/**
 * @brief Scalar implementation for BHWC maxpool,
 * MaxpoolScalarBHWC<24,24,12,12,1,6,2,2,2,2> total = 10758
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
class MaxpoolScalarBHWC {
  public:
    void filter(
      input_window<TT>* in,      // BHWC (1x24x24x6)
      output_stream<TT>* out     // BPQC (1x12x12x6)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(MaxpoolScalarBHWC::filter);
    }
};


/**
 * @brief Scalar implementation for BCHW maxpool,
 * MaxpoolScalarBCHW::filter<24,32,16,1,6> total = 19174
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
class MaxpoolScalarBCHW {
  public:
    void filter(
      input_window<TT>* in,      // BCHW (1x6x24x24)
      output_stream<TT>* out     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(MaxpoolScalarBCHW::filter);
    }
};


/**
 * @brief Vector implementation for float BCHW maxpool with 2x2 kernel,
 * requires INP_W%8==0, OUT_W%4==0, KH==KW==2, TT==float, 
 * Maxpool2x2FloatBCHW<24,24,12,12,1,6,2,2,2,2> total = 1977
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
class Maxpool2x2FloatBCHW {
  public:
    void filter(
      input_window<float>* in_window,      // BCHW (1x6x24x24)
      output_stream<float>* out_window     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      static_assert(INP_W % 8 == 0);
      static_assert(OUT_W % 4 == 0);
      static_assert(KH == 2);
      static_assert(KW == 2);
      static_assert((std::is_same<TT, float>::value));
      REGISTER_FUNCTION(Maxpool2x2FloatBCHW::filter);
    }
};


/**
 * @brief Vector implementation for int8 BCHW maxpool with 2x2 kernel,
 * requires INP_W%16==0, OUT_W%8==0, KH==KW==2, TT==int8_t, 
 * Maxpool2x2Int8BCHW::filter<24,32,16,1,6> total = 973
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
class Maxpool2x2Int8BCHW {
    static constexpr int RUN_16CHUNK = INP_W % 32 != 0;
  public:
    void filter(
      input_window<TT>* in_window,      // BCHW (1x6x24x24)
      output_stream<TT>* out_window     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      static_assert(INP_W % 16 == 0);
      static_assert(OUT_W % 8 == 0);
      static_assert(KH == 2);
      static_assert(KW == 2);
      static_assert((std::is_same<TT, int8_t>::value) || (std::is_same<TT, uint8_t>::value));
      REGISTER_FUNCTION(Maxpool2x2Int8BCHW::filter);
    }
};


/**
 * @brief Scalar implementation for BCHW avgpool,
 * AvgpoolScalarBCHW<24,24,12,12,1,6,2,2,2,2> total = 19575
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
class AvgpoolScalarBCHW {
  public:
    void filter(
      input_window<TT>* in,      // BCHW (1x6x24x24)
      output_stream<TT>* out     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(AvgpoolScalarBCHW::filter);
    }
};
/** @}*/


#endif // POOL_H_
