# import sys
# print(sys.version)

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("GPU Info: ", tf.config.list_physical_devices('GPU'))

import os
cuda_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cuda_kernels')
os.makedirs(cuda_dir, exist_ok=True)