# cuda_ops.py
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Load CUDA module
with open('path/to/cuda/conv_kernels.cu') as f:
    cuda_source = f.read()
    
mod = SourceModule(cuda_source)

def conv2d_forward(input, weights, bias, stride=1, padding=0):
    """
    Wrapper for the CUDA convolution kernel
    
    Args:
        input: numpy array of shape [N, C_in, H_in, W_in]
        weights: numpy array of shape [C_out, C_in, K_h, K_w]
        bias: numpy array of shape [C_out]
        stride: convolution stride
        padding: padding size
        
    Returns:
        output: numpy array of shape [N, C_out, H_out, W_out]
    """
    # Get input dimensions
    N, C_in, H_in, W_in = input.shape
    C_out, _, K_h, K_w = weights.shape
    
    # Calculate output dimensions
    H_out = (H_in + 2 * padding - K_h) // stride + 1
    W_out = (W_in + 2 * padding - K_w) // stride + 1
    
    # Allocate memory for output
    output = np.empty((N, C_out, H_out, W_out), dtype=np.float32)
    
    # Convert to contiguous arrays in C order
    input_c = np.ascontiguousarray(input.astype(np.float32))
    weights_c = np.ascontiguousarray(weights.astype(np.float32))
    bias_c = np.ascontiguousarray(bias.astype(np.float32))
    output_c = np.ascontiguousarray(output)
    
    # Get CUDA function
    func = mod.get_function("conv2d_forward")
    
    # Set up grid and block dimensions
    grid = (N, C_out, H_out * W_out)
    
    # Call kernel
    func(
        cuda.In(input_c), cuda.In(weights_c), cuda.In(bias_c), cuda.Out(output_c),
        np.int32(N), np.int32(C_in), np.int32(H_in), np.int32(W_in),
        np.int32(C_out), np.int32(K_h), np.int32(K_w), np.int32(padding), np.int32(stride),
        grid=(N, C_out, H_out * W_out), block=(1, 1, 1)
    )
    
    return output_c