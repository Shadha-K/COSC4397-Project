// pool_kernels.cu
#include <float.h>
#include <cfloat>

__global__ void max_pool_forward(
    const float* input,  // [N, C, H_in, W_in]
    float* output,  // [N, C, H_out, W_out]
    int N, int C, int H_in, int W_in,
    int pool_size, int stride) {
    
    // Calculate output dimensions
    int H_out = (H_in - pool_size) / stride + 1;
    int W_out = (W_in - pool_size) / stride + 1;
    
    // Get thread indices
    int n = blockIdx.x;  // Batch index
    int c = blockIdx.y;  // Channel index
    int h_out = blockIdx.z / W_out;  // Output height index
    int w_out = blockIdx.z % W_out;  // Output width index
    
    // Check if within bounds
    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out)
        return;
    
    // Compute max pooling
    float max_val = -FLT_MAX;
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int h_in = h_out * stride + ph;
            int w_in = w_out * stride + pw;
            
            if (h_in < H_in && w_in < W_in) {
                float val = input[((n * C + c) * H_in + h_in) * W_in + w_in];
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    // Write output
    output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
}