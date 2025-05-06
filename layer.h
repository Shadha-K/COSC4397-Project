#ifndef _LAYER_H_
#define _LAYER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <math.h>

// Learning rate
#define dt 0.01

class Layer {
public:
    int M, N, O;
    float *output;
    float *preact;

    float *bias;
    float *weight;

    float *d_output;
    float *d_preact;
    float *d_weight;

    Layer(int M, int N, int O);
    ~Layer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
};

// Device functions for activation
__device__ float relu_function(float v);
__device__ float softmax_function(float v);

// Forward pass kernels
__global__ void apply_relu_function(float *input, float *output, const int N);
__global__ void apply_softmax_function(float *input, float *output, const int N);
__global__ void fp_preact_c1(float input[32][32][3], float preact[6][28][28], float weight[6][5][5][3]);
__global__ void fp_bias_c1(float preact[6][28][28], float bias[6]);
__global__ void fp_preact_s1(float input[6][28][28], float preact[6][7][7], float weight[1][4][4]);
__global__ void fp_bias_s1(float preact[6][7][7], float bias[1]);
__global__ void fp_preact_f(float input[6][7][7], float preact[10], float weight[10][6][7][7]);
__global__ void fp_bias_f(float preact[10], float bias[10]);

// Backpropagation kernels
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);
__global__ void bp_weight_f(float d_weight[10][6][7][7], float d_preact[10], float p_output[6][7][7]);
__global__ void bp_bias_f(float bias[10], float d_preact[10]);
__global__ void bp_output_s1(float d_output[6][7][7], float n_weight[10][6][7][7], float nd_preact[10]);
__global__ void bp_preact_s1(float d_preact[6][7][7], float d_output[6][7][7], float preact[6][7][7]);
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][7][7], float p_output[6][28][28]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][7][7]);
__global__ void bp_output_c1(float d_output[6][28][28], float n_weight[1][4][4], float nd_preact[6][7][7]);
__global__ void bp_preact_c1(float d_preact[6][28][28], float d_output[6][28][28], float preact[6][28][28]);
__global__ void bp_weight_c1(float d_weight[6][5][5][3], float d_preact[6][28][28], float p_output[32][32][3]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][28][28]);

#endif