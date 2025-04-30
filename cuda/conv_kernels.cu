// conv_kernels.cu
#include <float.h>
#include <cfloat>

__global__ void conv2d_forward(
    const float* input,  // [N, C_in, H_in, W_in]
    const float* weights,  // [C_out, C_in, K_h, K_w]
    const float* bias,  // [C_out]
    float* output,  // [N, C_out, H_out, W_out]
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w, int pad, int stride) {
    
    // Calculate output dimensions
    int H_out = (H_in + 2 * pad - K_h) / stride + 1;
    int W_out = (W_in + 2 * pad - K_w) / stride + 1;
    
    // Get thread indices
    int n = blockIdx.x;  // Batch index
    int c_out = blockIdx.y;  // Output channel index
    int h_out = blockIdx.z / W_out;  // Output height index
    int w_out = blockIdx.z % W_out;  // Output width index
    
    // Check if within bounds
    if (n >= N || c_out >= C_out || h_out >= H_out || w_out >= W_out)
        return;
    
    // Compute convolution
    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < K_h; kh++) {
            for (int kw = 0; kw < K_w; kw++) {
                int h_in = h_out * stride - pad + kh;
                int w_in = w_out * stride - pad + kw;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    float input_val = input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
                    float weight_val = weights[((c_out * C_in + c_in) * K_h + kh) * K_w + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    // Write output
    output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = sum;
}