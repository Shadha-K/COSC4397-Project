// fc_kernels.cu
#include <float.h>
#include <cfloat>

__global__ void fc_forward(
    const float* input,  // [N, in_features]
    const float* weights,  // [out_features, in_features]
    const float* bias,  // [out_features]
    float* output,  // [N, out_features]
    int N, int in_features, int out_features) {
    
    // Get thread indices
    int n = blockIdx.x;  // Batch sample index
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;  // Output feature index
    
    // Check if within bounds
    if (n >= N || out_idx >= out_features)
        return;
    
    // Compute matrix multiplication
    float sum = 0.0f;
    for (int in_idx = 0; in_idx < in_features; in_idx++) {
        sum += input[n * in_features + in_idx] * weights[out_idx * in_features + in_idx];
    }
    
    // Add bias
    sum += bias[out_idx];
    
    // Write output
    output[n * out_features + out_idx] = sum;
}