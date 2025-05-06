"""
Simple test script to diagnose CUDA kernel issues.
This isolates just the CUDA functionality to identify where the error occurs.
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Define a minimal CUDA kernel for testing
test_kernel_source = """
__global__ void test_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;  // Just double the input
    }
}
"""

# Compile the test kernel
test_module = SourceModule(test_kernel_source)
test_kernel = test_module.get_function("test_kernel")

# Create test data
size = 1024
input_data = np.random.randn(size).astype(np.float32)
output_data = np.empty_like(input_data)

# Allocate device memory
d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(output_data.nbytes)

# Copy data to device
cuda.memcpy_htod(d_input, input_data)

# Set up grid and block dimensions
threads_per_block = 256
blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

try:
    # Call kernel
    test_kernel(
        d_input, d_output, np.int32(size),
        grid=(blocks_per_grid, 1, 1), 
        block=(threads_per_block, 1, 1)
    )
    
    # Copy result back to host
    cuda.memcpy_dtoh(output_data, d_output)
    
    # Verify result
    expected_output = input_data * 2.0
    max_diff = np.max(np.abs(output_data - expected_output))
    
    print("Test successful!")
    print(f"Maximum difference: {max_diff:.6f}")
    
except Exception as e:
    print(f"Error running CUDA kernel: {e}")
    print("Please check your CUDA installation and device compatibility.")

# Try a different block configuration if you're still seeing errors
try:
    print("\nTrying alternative block configuration...")
    test_kernel(
        d_input, d_output, np.int32(size),
        grid=(blocks_per_grid, 1), 
        block=(threads_per_block, 1)
    )
    
    cuda.memcpy_dtoh(output_data, d_output)
    print("Alternative configuration successful!")
    
except Exception as e:
    print(f"Error with alternative configuration: {e}")