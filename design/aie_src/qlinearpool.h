#ifndef QLINEARPOOL_H_
#define QLINEARPOOL_H_

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
 * @brief Scalar implementation for BCHW avgpool,
 * QLinearAvgpoolScalarBCHW::filter<24,32,16,1,6> total = 15766
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
class QLinearAvgpoolScalarBCHW {
  private:
    static constexpr int K = INP_H/OUT_H;
    float in_scale;
    float out_scale;
    int8_t in_zero;
    int8_t out_zero;
    
  public:
    QLinearAvgpoolScalarBCHW (
      float in_scale,
      float out_scale,
      int8_t in_zero,
      int8_t out_zero
    ): in_scale(in_scale), out_scale(out_scale), in_zero(in_zero), out_zero(out_zero) {}

    void filter(
      input_window<TT>* in,      // BCHW (1x6x24x24)
      output_window<TT>* out     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(QLinearAvgpoolScalarBCHW::filter);
    }
};
/** @}*/


#endif // QLINEARPOOL_H_
