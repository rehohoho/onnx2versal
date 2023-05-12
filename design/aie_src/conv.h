#ifndef CONV_H_
#define CONV_H_

#include <adf.h>


template <int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalarGmemParamBHWC {
  public:
    void filter(
      input_window<float>* in,      // BHWC (1x28x28x1)
      input_window<float>* weight,  // MKKC (6x5x5x1)
      input_window<float>* bias,    // M    (6)
      output_window<float>* out     // BHWM (1x24x24x6)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalarGmemParamBHWC::filter);
    }
};


template <int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalarBHWC {

  private:
    alignas(32) float (&weights)[M*K*K*C];
    alignas(32) float (&bias)[M];

  public:
    ConvReluScalarBHWC(
      float (&w)[M*K*K*C], // only accepts reference to MKKC array
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BHWC (1x28x28x1)
      output_window<float>* out     // BHWM (1x24x24x6)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalarBHWC::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


template <int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalarBCHW {

  private:
    alignas(32) float (&weights)[M*K*K*C];
    alignas(32) float (&bias)[M];

  public:
    ConvReluScalarBCHW(
      float (&w)[M*K*K*C], // only accepts reference to MKKC array
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BHWC (1x28x28x1)
      output_window<float>* out     // BHWM (1x24x24x6)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalarBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


template <int INP_W, int OUT_W, int B, int C, int M, int _K_notused>
class Conv5x5ReluBCHW {

  private:
    static const int K = 5;
    alignas(32) float (&weights)[M*C*K*K];
    alignas(32) float (&bias)[M];

  public:
    Conv5x5ReluBCHW(
      float (&w)[M*C*K*K], // only accepts reference to MKKC array
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(Conv5x5ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


template <int INP_W, int OUT_W, int B, int C, int M, int _K_notused>
class Conv5x5on8ReluBCHW {

  private:
    static const int K = 5;
    alignas(32) float (&weights)[M*C*K*8];
    alignas(32) float (&bias)[M];

  public:
    Conv5x5on8ReluBCHW(
      float (&w)[M*C*K*8], // only accepts reference to MKKC array
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(Conv5x5on8ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


#endif // CONV_H_
