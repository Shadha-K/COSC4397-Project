"""
CUDA-accelerated LeNet implementation for CIFAR-10 classification.
This implementation uses custom CUDA kernels for convolution, pooling, and fully connected layers.
"""

import os
import time
import numpy as np
from typing import Tuple, List, Optional

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Load CUDA kernel sources
def load_kernel_source(filename: str) -> str:
    """Load CUDA kernel source from file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_dir = os.path.join(current_dir, 'cuda')
    with open(os.path.join(cuda_dir, filename), 'r') as f:
        return f.read()

# Compile CUDA kernels
conv_source = load_kernel_source('conv_kernels.cu')
pool_source = load_kernel_source('pool_kernels.cu')
fc_source = load_kernel_source('fc_kernels.cu')

conv_module = SourceModule(conv_source)
pool_module = SourceModule(pool_source)
fc_module = SourceModule("""
#include <float.h>  // For FLT_MAX

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

__global__ void relu_forward(
    const float* input,
    float* output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

__global__ void softmax_forward(
    const float* input,
    float* output,
    int batch_size,
    int num_classes) {
    
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size)
        return;
        
    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < num_classes; i++) {
        max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        float exp_val = expf(input[batch_idx * num_classes + i] - max_val);
        output[batch_idx * num_classes + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < num_classes; i++) {
        output[batch_idx * num_classes + i] /= sum;
    }
}
""")

# Get CUDA function references
conv2d_kernel = conv_module.get_function("conv2d_forward")
max_pool_kernel = pool_module.get_function("max_pool_forward")
fc_kernel = fc_module.get_function("fc_forward")
relu_kernel = fc_module.get_function("relu_forward")
softmax_kernel = fc_module.get_function("softmax_forward")

class CudaLeNet:
    """
    LeNet CNN implementation using custom CUDA kernels
    Architecture:
    1. Conv1: 3 -> 6 channels, 5x5 kernel
    2. ReLU
    3. MaxPool: 2x2, stride 2
    4. Conv2: 6 -> 16 channels, 5x5 kernel
    5. ReLU
    6. MaxPool: 2x2, stride 2
    7. FC1: 16*5*5 -> 120
    8. ReLU
    9. FC2: 120 -> 84
    10. ReLU
    11. FC3: 84 -> 10 (output)
    12. Softmax
    """
    
    def __init__(self, use_cuda: bool = True):
        """
        Initialize LeNet model with random weights
        
        Args:
            use_cuda: Whether to use CUDA for forward pass
        """
        self.use_cuda = use_cuda
        
        # Layer 1: Conv (3 -> 6, 5x5)
        self.conv1_weights = np.random.randn(6, 3, 5, 5).astype(np.float32) * np.sqrt(2.0 / (3 * 5 * 5))
        self.conv1_bias = np.zeros(6, dtype=np.float32)
        
        # Layer 2: Conv (6 -> 16, 5x5)
        self.conv2_weights = np.random.randn(16, 6, 5, 5).astype(np.float32) * np.sqrt(2.0 / (6 * 5 * 5))
        self.conv2_bias = np.zeros(16, dtype=np.float32)
        
        # Layer 3: FC (16*5*5 -> 120)
        self.fc1_weights = np.random.randn(120, 16 * 5 * 5).astype(np.float32) * np.sqrt(2.0 / (16 * 5 * 5))
        self.fc1_bias = np.zeros(120, dtype=np.float32)
        
        # Layer 4: FC (120 -> 84)
        self.fc2_weights = np.random.randn(84, 120).astype(np.float32) * np.sqrt(2.0 / 120)
        self.fc2_bias = np.zeros(84, dtype=np.float32)
        
        # Layer 5: FC (84 -> 10)
        self.fc3_weights = np.random.randn(10, 84).astype(np.float32) * np.sqrt(2.0 / 84)
        self.fc3_bias = np.zeros(10, dtype=np.float32)
        
        # Initialize device memory for parameters
        if use_cuda:
            self._init_cuda_memory()
    
    def _init_cuda_memory(self):
        """Initialize CUDA memory for model parameters"""
        # Allocate and copy weights to GPU
        self.d_conv1_weights = cuda.mem_alloc(self.conv1_weights.nbytes)
        self.d_conv1_bias = cuda.mem_alloc(self.conv1_bias.nbytes)
        cuda.memcpy_htod(self.d_conv1_weights, self.conv1_weights)
        cuda.memcpy_htod(self.d_conv1_bias, self.conv1_bias)
        
        self.d_conv2_weights = cuda.mem_alloc(self.conv2_weights.nbytes)
        self.d_conv2_bias = cuda.mem_alloc(self.conv2_bias.nbytes)
        cuda.memcpy_htod(self.d_conv2_weights, self.conv2_weights)
        cuda.memcpy_htod(self.d_conv2_bias, self.conv2_bias)
        
        self.d_fc1_weights = cuda.mem_alloc(self.fc1_weights.nbytes)
        self.d_fc1_bias = cuda.mem_alloc(self.fc1_bias.nbytes)
        cuda.memcpy_htod(self.d_fc1_weights, self.fc1_weights)
        cuda.memcpy_htod(self.d_fc1_bias, self.fc1_bias)
        
        self.d_fc2_weights = cuda.mem_alloc(self.fc2_weights.nbytes)
        self.d_fc2_bias = cuda.mem_alloc(self.fc2_bias.nbytes)
        cuda.memcpy_htod(self.d_fc2_weights, self.fc2_weights)
        cuda.memcpy_htod(self.d_fc2_bias, self.fc2_bias)
        
        self.d_fc3_weights = cuda.mem_alloc(self.fc3_weights.nbytes)
        self.d_fc3_bias = cuda.mem_alloc(self.fc3_bias.nbytes)
        cuda.memcpy_htod(self.d_fc3_weights, self.fc3_weights)
        cuda.memcpy_htod(self.d_fc3_bias, self.fc3_bias)
    
    def _conv2d_forward(self, input_data: np.ndarray, weights, bias, 
                   stride: int = 1, padding: int = 0) -> np.ndarray:
        """
        Convolution operation with custom CUDA kernel
        
        Args:
            input_data: Input data [N, C_in, H_in, W_in]
            weights: Convolution weights [C_out, C_in, K_h, K_w] or DeviceAllocation
            bias: Bias [C_out] or DeviceAllocation
            stride: Stride for convolution
            padding: Padding for convolution
            
        Returns:
            Output feature map [N, C_out, H_out, W_out]
        """
        # Get dimensions
        N, C_in, H_in, W_in = input_data.shape
        
        # Handle both DeviceAllocation and numpy arrays
        if isinstance(weights, cuda.DeviceAllocation):
            # Determine which layer's weights are being used and get the corresponding shape
            if weights == self.d_conv1_weights:
                C_out, C_in_w, K_h, K_w = self.conv1_weights.shape
            elif weights == self.d_conv2_weights:
                C_out, C_in_w, K_h, K_w = self.conv2_weights.shape
            else:
                # Handle other possible device weights
                raise ValueError("Unknown device weights")
        else:
            # For numpy arrays, get shape directly
            C_out, C_in_w, K_h, K_w = weights.shape
        
        # Calculate output dimensions
        H_out = (H_in + 2 * padding - K_h) // stride + 1
        W_out = (W_in + 2 * padding - K_w) // stride + 1
        
        # Allocate output array
        output = np.empty((N, C_out, H_out, W_out), dtype=np.float32)
        
        if self.use_cuda:
            # Allocate device memory
            d_input = cuda.mem_alloc(input_data.nbytes)
            d_output = cuda.mem_alloc(output.nbytes)
            
            # Copy data to device
            cuda.memcpy_htod(d_input, input_data)
            
            # Set up grid and block dimensions
            # Using 3D grid: (batch_size, output_channels, height*width)
            grid = (N, C_out, H_out * W_out)
            block = (1, 1, 1)
            
            # Call kernel
            if isinstance(weights, np.ndarray):
                # Use weights from arguments
                d_weights = cuda.mem_alloc(weights.nbytes)
                d_bias = cuda.mem_alloc(bias.nbytes)
                cuda.memcpy_htod(d_weights, weights)
                cuda.memcpy_htod(d_bias, bias)
                
                conv2d_kernel(
                    d_input, d_weights, d_bias, d_output,
                    np.int32(N), np.int32(C_in), np.int32(H_in), np.int32(W_in),
                    np.int32(C_out), np.int32(K_h), np.int32(K_w), 
                    np.int32(padding), np.int32(stride),
                    grid=grid, block=block
                )
            else:
                # Use weights from model (device memory)
                conv2d_kernel(
                    d_input, weights, bias, d_output,
                    np.int32(N), np.int32(C_in), np.int32(H_in), np.int32(W_in),
                    np.int32(C_out), np.int32(K_h), np.int32(K_w), 
                    np.int32(padding), np.int32(stride),
                    grid=grid, block=block
                )
            
            # Copy result back to host
            cuda.memcpy_dtoh(output, d_output)
            
            return output
        else:
            # CPU implementation (for testing)
            for n in range(N):
                for c_out in range(C_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            sum_val = 0.0
                            for c_in in range(C_in):
                                for kh in range(K_h):
                                    for kw in range(K_w):
                                        h_in = h_out * stride - padding + kh
                                        w_in = w_out * stride - padding + kw
                                        
                                        if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                            sum_val += input_data[n, c_in, h_in, w_in] * weights[c_out, c_in, kh, kw]
                            
                            output[n, c_out, h_out, w_out] = sum_val + bias[c_out]
        
        return output
    
    def _max_pool_forward(self, input_data: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
        """
        Max pooling operation with custom CUDA kernel
        
        Args:
            input_data: Input data [N, C, H_in, W_in]
            pool_size: Size of pooling window
            stride: Stride for pooling
            
        Returns:
            Output feature map [N, C, H_out, W_out]
        """
        # Get dimensions
        N, C, H_in, W_in = input_data.shape
        
        # Calculate output dimensions
        H_out = (H_in - pool_size) // stride + 1
        W_out = (W_in - pool_size) // stride + 1
        
        # Allocate output array
        output = np.empty((N, C, H_out, W_out), dtype=np.float32)
        
        if self.use_cuda:
            # Allocate device memory
            d_input = cuda.mem_alloc(input_data.nbytes)
            d_output = cuda.mem_alloc(output.nbytes)
            
            # Copy data to device
            cuda.memcpy_htod(d_input, input_data)
            
            # Set up grid and block dimensions
            # Using 3D grid: (batch_size, channels, height*width)
            grid = (N, C, H_out * W_out)
            block = (1, 1, 1)
            
            # Call kernel
            max_pool_kernel(
                d_input, d_output,
                np.int32(N), np.int32(C), np.int32(H_in), np.int32(W_in),
                np.int32(pool_size), np.int32(stride),
                grid=grid, block=block
            )
            
            # Copy result back to host
            cuda.memcpy_dtoh(output, d_output)
            
            return output
        else:
            # CPU implementation (for testing)
            for n in range(N):
                for c in range(C):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            max_val = float('-inf')
                            for ph in range(pool_size):
                                for pw in range(pool_size):
                                    h_in = h_out * stride + ph
                                    w_in = w_out * stride + pw
                                    
                                    if h_in < H_in and w_in < W_in:
                                        max_val = max(max_val, input_data[n, c, h_in, w_in])
                            
                            output[n, c, h_out, w_out] = max_val
            
            return output
    
    def _fc_forward(self, input_data: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """
        Fully connected layer with custom CUDA kernel
        
        Args:
            input_data: Input data [N, in_features]
            weights: Weights [out_features, in_features]
            bias: Bias [out_features]
            
        Returns:
            Output [N, out_features]
        """
        # Get input dimensions
        N, in_features = input_data.shape

        # If weights are on device, get shape from corresponding CPU copy
        if isinstance(weights, cuda.DeviceAllocation):
            if weights == self.d_fc1_weights:
                out_features = self.fc1_weights.shape[0]
            elif weights == self.d_fc2_weights:
                out_features = self.fc2_weights.shape[0]
            elif weights == self.d_fc3_weights:
                out_features = self.fc3_weights.shape[0]
            else:
                raise ValueError("Unknown device weights")
        else:
            out_features = weights.shape[0]
        
        # Allocate output
        output = np.empty((N, out_features), dtype=np.float32)

        
        if self.use_cuda:
            # Allocate device memory
            d_input = cuda.mem_alloc(input_data.nbytes)
            d_output = cuda.mem_alloc(output.nbytes)
            
            # Copy data to device
            cuda.memcpy_htod(d_input, input_data)
            
            # Set up grid and block dimensions
            # Each thread computes one output element
            threads_per_block = min(32, out_features)
            blocks_per_grid_y = (out_features + threads_per_block - 1) // threads_per_block
            grid = (N, blocks_per_grid_y)
            block = (threads_per_block, 1, 1)
            
            # Call kernel
            if isinstance(weights, np.ndarray):
                # Use weights from arguments
                d_weights = cuda.mem_alloc(weights.nbytes)
                d_bias = cuda.mem_alloc(bias.nbytes)
                cuda.memcpy_htod(d_weights, weights)
                cuda.memcpy_htod(d_bias, bias)
                
                fc_kernel(
                    d_input, d_weights, d_bias, d_output,
                    np.int32(N), np.int32(in_features), np.int32(out_features),
                    grid=grid, block=block
                )
            else:
                # Use weights from model (device memory)
                fc_kernel(
                    d_input, weights, bias, d_output,
                    np.int32(N), np.int32(in_features), np.int32(out_features),
                    grid=grid, block=block
                )
            
            # Copy result back to host
            cuda.memcpy_dtoh(output, d_output)
            
            return output
        else:
            # CPU implementation (for testing)
            for n in range(N):
                for out_idx in range(out_features):
                    sum_val = 0.0
                    for in_idx in range(in_features):
                        sum_val += input_data[n, in_idx] * weights[out_idx, in_idx]
                    
                    output[n, out_idx] = sum_val + bias[out_idx]
            
            return output
    
    def _relu_forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        ReLU activation function with CUDA kernel
        
        Args:
            input_data: Input data of any shape
            
        Returns:
            Output with same shape as input
        """
        # Get total size
        size = input_data.size
        shape = input_data.shape
        
        # Flatten input for processing
        input_flat = input_data.reshape(-1).astype(np.float32)
        
        # Allocate output array
        output_flat = np.empty_like(input_flat)
        
        if self.use_cuda:
            # Allocate device memory
            d_input = cuda.mem_alloc(input_flat.nbytes)
            d_output = cuda.mem_alloc(output_flat.nbytes)
            
            # Copy data to device
            cuda.memcpy_htod(d_input, input_flat)
            
            # Set up grid and block dimensions
            threads_per_block = 256
            blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
            
            # Call kernel
            relu_kernel(
                d_input, d_output, np.int32(size),
                grid=(blocks_per_grid, 1, 1), block=(threads_per_block, 1, 1)
            )
            
            # Copy result back to host
            cuda.memcpy_dtoh(output_flat, d_output)
            
            return output_flat.reshape(shape)
        else:
            # CPU implementation (for testing)
            return np.maximum(input_data, 0)
    
    def _softmax_forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Softmax activation function with CUDA kernel
        
        Args:
            input_data: Input data [N, num_classes]
            
        Returns:
            Output [N, num_classes] with softmax applied along axis 1
        """
        # Get dimensions
        N, num_classes = input_data.shape
        
        # Allocate output array
        output = np.empty_like(input_data)
        
        if self.use_cuda:
            # Allocate device memory
            d_input = cuda.mem_alloc(input_data.nbytes)
            d_output = cuda.mem_alloc(output.nbytes)
            
            # Copy data to device
            cuda.memcpy_htod(d_input, input_data)
            
            # Set up grid and block dimensions
            grid = (N, 1, 1)
            block = (1, 1, 1)
            
            # Call kernel
            softmax_kernel(
                d_input, d_output, np.int32(N), np.int32(num_classes),
                grid=grid, block=block
            )
            
            # Copy result back to host
            cuda.memcpy_dtoh(output, d_output)
            
            return output
        else:
            # CPU implementation (for testing)
            for n in range(N):
                # Shift inputs for numerical stability
                shifted = input_data[n] - np.max(input_data[n])
                exp_vals = np.exp(shifted)
                output[n] = exp_vals / np.sum(exp_vals)
            
            return output
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network
        
        Args:
            x: Input batch of images [N, C, H, W]
            
        Returns:
            Output predictions [N, num_classes]
        """
        # Make sure input is float32
        x = x.astype(np.float32)
        
        # Layer 1: Conv + ReLU + Pool
        if self.use_cuda:
            conv1 = self._conv2d_forward(x, self.d_conv1_weights, self.d_conv1_bias)
        else:
            conv1 = self._conv2d_forward(x, self.conv1_weights, self.conv1_bias)
        relu1 = self._relu_forward(conv1)
        pool1 = self._max_pool_forward(relu1)
        
        # Layer 2: Conv + ReLU + Pool
        if self.use_cuda:
            conv2 = self._conv2d_forward(pool1, self.d_conv2_weights, self.d_conv2_bias)
        else:
            conv2 = self._conv2d_forward(pool1, self.conv2_weights, self.conv2_bias)
        relu2 = self._relu_forward(conv2)
        pool2 = self._max_pool_forward(relu2)
        
        # Flatten
        N = pool2.shape[0]
        flatten = pool2.reshape(N, -1)
        
        # Layer 3: FC + ReLU
        if self.use_cuda:
            fc1 = self._fc_forward(flatten, self.d_fc1_weights, self.d_fc1_bias)
        else:
            fc1 = self._fc_forward(flatten, self.fc1_weights, self.fc1_bias)
        relu3 = self._relu_forward(fc1)
        
        # Layer 4: FC + ReLU
        if self.use_cuda:
            fc2 = self._fc_forward(relu3, self.d_fc2_weights, self.d_fc2_bias)
        else:
            fc2 = self._fc_forward(relu3, self.fc2_weights, self.fc2_bias)
        relu4 = self._relu_forward(fc2)
        
        # Layer 5: FC + Softmax
        if self.use_cuda:
            fc3 = self._fc_forward(relu4, self.d_fc3_weights, self.d_fc3_bias)
        else:
            fc3 = self._fc_forward(relu4, self.fc3_weights, self.fc3_bias)
        out = self._softmax_forward(fc3)
        
        return out
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input images
        
        Args:
            x: Input batch of images [N, C, H, W]
            
        Returns:
            Predicted class labels [N]
        """
        outputs = self.forward(x)
        return np.argmax(outputs, axis=1)
    
    def free_memory(self):
        """Free CUDA memory"""
        if hasattr(self, 'd_conv1_weights'):
            # Free all device memory
            for attr_name in dir(self):
                if attr_name.startswith('d_') and isinstance(getattr(self, attr_name), cuda.DeviceAllocation):
                    getattr(self, attr_name).free()

# Example usage
if __name__ == "__main__":
    # Create model
    model = CudaLeNet(use_cuda=True)
    
    # Generate random input (batch of 4 CIFAR-10 images)
    x = np.random.randn(4, 3, 32, 32).astype(np.float32)
    
    # Time forward pass
    start_time = time.time()
    outputs = model.forward(x)
    cuda_time = time.time() - start_time
    
    print(f"CUDA forward pass time: {cuda_time:.6f} seconds")
    print(f"Output shape: {outputs.shape}")
    
    # Get predictions
    predictions = model.predict(x)
    print(f"Predictions: {predictions}")
    
    # Compare with CPU implementation
    model_cpu = CudaLeNet(use_cuda=False)
    # Copy weights to ensure fair comparison
    model_cpu.conv1_weights = model.conv1_weights.copy()
    model_cpu.conv1_bias = model.conv1_bias.copy()
    model_cpu.conv2_weights = model.conv2_weights.copy()
    model_cpu.conv2_bias = model.conv2_bias.copy()
    model_cpu.fc1_weights = model.fc1_weights.copy()
    model_cpu.fc1_bias = model.fc1_bias.copy()
    model_cpu.fc2_weights = model.fc2_weights.copy()
    model_cpu.fc2_bias = model.fc2_bias.copy()
    model_cpu.fc3_weights = model.fc3_weights.copy()
    model_cpu.fc3_bias = model.fc3_bias.copy()
    
    start_time = time.time()
    outputs_cpu = model_cpu.forward(x)
    cpu_time = time.time() - start_time
    
    print(f"CPU forward pass time: {cpu_time:.6f} seconds")
    print(f"Speedup: {cpu_time / cuda_time:.2f}x")
    
    # Check for correctness
    max_diff = np.max(np.abs(outputs - outputs_cpu))
    print(f"Maximum difference between CUDA and CPU: {max_diff:.6f}")
    
    # Free CUDA memory
    model.free_memory()