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
 * MaxpoolScalarBHWC::filter<24,32,16,1,6> total = 7673
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class MaxpoolScalarBHWC {
  public:
    void filter(
      input_window<TT>* in,      // BHWC (1x24x24x6)
      output_window<TT>* out     // BPQC (1x12x12x6)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(MaxpoolScalarBHWC::filter);
    }
};


/**
 * @brief Scalar implementation for BCHW maxpool,
 * MaxpoolScalarBCHW::filter<24,32,16,1,6> total = 11302
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class MaxpoolScalarBCHW {
  public:
    void filter(
      input_window<TT>* in,      // BCHW (1x6x24x24)
      output_window<TT>* out     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(MaxpoolScalarBCHW::filter);
    }
};


/**
 * @brief Vector implementation for float BCHW maxpool with 2x2 kernel,
 * requires INP_W%8==0, OUT_W%4==0, KH==KW==2, TT==float, 
 * Maxpool2x2FloatBCHW::filter<24,32,16,1,6> total = 901
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class Maxpool2x2FloatBCHW {
  public:
    void filter(
      input_window<float>* in_window,      // BCHW (1x6x24x24)
      output_window<float>* out_window     // BCPQ (1x6x12x12)
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
 * Maxpool2x2Int8BCHW::filter<24,32,16,1,6> total = 324
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class Maxpool2x2Int8BCHW {
    static constexpr int RUN_16CHUNK = INP_W % 32 != 0;
  public:
    void filter(
      input_window<int8_t>* in_window,      // BCHW (1x6x24x24)
      output_window<int8_t>* out_window     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      static_assert(INP_W % 16 == 0);
      static_assert(OUT_W % 8 == 0);
      static_assert(KH == 2);
      static_assert(KW == 2);
      static_assert((std::is_same<TT, int8_t>::value));
      REGISTER_FUNCTION(Maxpool2x2Int8BCHW::filter);
    }
};


/**
 * @brief Scalar implementation for BCHW avgpool,
 * AvgpoolScalarBCHW::filter<24,32,16,1,6> total = 15766
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class AvgpoolScalarBCHW {
  public:
    void filter(
      input_window<TT>* in,      // BCHW (1x6x24x24)
      output_window<TT>* out     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(AvgpoolScalarBCHW::filter);
    }
};
/** @}*/


#endif // POOL_H_
