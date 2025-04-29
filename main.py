import sys
print(sys.version)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Info: ", tf.config.list_physical_devices('GPU'))