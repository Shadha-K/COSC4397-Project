# data_loader.py
import numpy as np
from tensorflow.keras.datasets import cifar10

def load_cifar10_subset(train_size=2500, test_size=500, normalize=True, seed=42):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Optional: Normalize to [-1, 1]
    if normalize:
        x_train = (x_train / 127.5) - 1
        x_test = (x_test / 127.5) - 1

    # Random subsampling
    np.random.seed(seed)
    train_indices = np.random.choice(x_train.shape[0], train_size, replace=False)
    test_indices = np.random.choice(x_test.shape[0], test_size, replace=False)
    
    x_train_sampled = x_train[train_indices]
    y_train_sampled = y_train[train_indices]
    x_test_sampled = x_test[test_indices]
    y_test_sampled = y_test[test_indices]

    return (x_train_sampled, y_train_sampled), (x_test_sampled, y_test_sampled)