# model.py
import numpy as np
from wrappers.cuda_ops import conv2d_forward, max_pool_forward, fc_forward
import pycuda.autoinit  # Initialize CUDA

class LeNet:
    def __init__(self):
        # Initialize model parameters
        self.conv1_weights = np.random.randn(6, 3, 5, 5).astype(np.float32) * 0.01
        self.conv1_bias = np.zeros(6, dtype=np.float32)
        
        self.conv2_weights = np.random.randn(16, 6, 5, 5).astype(np.float32) * 0.01
        self.conv2_bias = np.zeros(16, dtype=np.float32)
        
        # For CIFAR-10 (32x32 images), output size after 2 convs and pools will be 5x5
        self.fc1_weights = np.random.randn(120, 16 * 5 * 5).astype(np.float32) * 0.01
        self.fc1_bias = np.zeros(120, dtype=np.float32)
        
        self.fc2_weights = np.random.randn(84, 120).astype(np.float32) * 0.01
        self.fc2_bias = np.zeros(84, dtype=np.float32)
        
        self.fc3_weights = np.random.randn(10, 84).astype(np.float32) * 0.01
        self.fc3_bias = np.zeros(10, dtype=np.float32)
    
    def forward(self, x):
        # Layer 1: Conv + ReLU + Pool
        conv1 = conv2d_forward(x, self.conv1_weights, self.conv1_bias, stride=1, padding=0)
        relu1 = np.maximum(conv1, 0)  # ReLU activation
        pool1 = max_pool_forward(relu1, pool_size=2, stride=2)
        
        # Layer 2: Conv + ReLU + Pool
        conv2 = conv2d_forward(pool1, self.conv2_weights, self.conv2_bias, stride=1, padding=0)
        relu2 = np.maximum(conv2, 0)  # ReLU activation
        pool2 = max_pool_forward(relu2, pool_size=2, stride=2)
        
        # Flatten
        flatten = pool2.reshape(pool2.shape[0], -1)
        
        # Layer 3: FC + ReLU
        fc1 = fc_forward(flatten, self.fc1_weights, self.fc1_bias)
        relu3 = np.maximum(fc1, 0)  # ReLU activation
        
        # Layer 4: FC + ReLU
        fc2 = fc_forward(relu3, self.fc2_weights, self.fc2_bias)
        relu4 = np.maximum(fc2, 0)  # ReLU activation
        
        # Layer 5: FC (output)
        fc3 = fc_forward(relu4, self.fc3_weights, self.fc3_bias)
        
        return fc3