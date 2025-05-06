import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
import os
# Force TensorFlow to use CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # Check for GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print(f"Using GPU: {physical_devices[0]}")
# else:
#     print("No GPU found, using CPU instead.")

# # Confirm CPU usage
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     print("GPU devices found but disabled. Forcing CPU usage.")
# else:
#     print("No GPU found, using CPU.")

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values to be in the range [-1, 1]
x_train = (x_train / 127.5) - 1
x_test = (x_test / 127.5) - 1

# Randomly sample 2,500 images from the training set
train_indices = np.random.choice(x_train.shape[0], 2500, replace=False)
x_train_sampled = x_train[train_indices]
y_train_sampled = y_train[train_indices]

# Randomly sample 500 images from the test set
test_indices = np.random.choice(x_test.shape[0], 500, replace=False)
x_test_sampled = x_test[test_indices]
y_test_sampled = y_test[test_indices]

print(f'Sampled train images shape: {x_train_sampled.shape}')
print(f'Sampled test images shape: {x_test_sampled.shape}')

# Randomly select 5 images from the training subset for visualization
sample_indices = np.random.choice(len(x_train_sampled), 5, replace=False)
x_sample = x_train_sampled[sample_indices]
y_sample = y_train_sampled[sample_indices]

# CIFAR-10 class labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Set up the plot
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

# Plot each image and its label
for i, ax in enumerate(axes):
    # Undo the normalization for display purposes (convert back to [0, 1] range)
    ax.imshow((x_sample[i] + 1) / 2)
    ax.set_title(labels[y_sample[i][0]])  # y_sample contains labels as integers
    ax.axis('off')  # Hide axes

# Save the plot instead of showing it
plt.savefig('sampled_images.png')
plt.close()

# Define the LeNet-5 architecture
def build_lenet5():
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3), padding='same'),
        layers.AveragePooling2D(pool_size=(2, 2)),

        # Convolutional Layer 2
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),

        # Flatten Layer
        layers.Flatten(),

        # Fully Connected Layer 1
        layers.Dense(120, activation='relu'),

        # Fully Connected Layer 2
        layers.Dense(84, activation='relu'),

        # Output Layer
        layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])
    return model

# Build the model
lenet5 = build_lenet5()

# Compile the model
lenet5.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# List to store histories
histories = []
labels = []

# Train and store the baseline model's history
history_baseline = lenet5.fit(
    x_train_sampled, y_train_sampled,
    epochs=20,
    batch_size=32,
    validation_data=(x_test_sampled, y_test_sampled),
    verbose=1
)

# Append the baseline history and label
histories.append(history_baseline)
labels.append("Baseline (No Regularization)")

# Extract loss and accuracy
# Access the history of the first training run in the `histories` list
train_loss = histories[0].history['loss']
train_accuracy = histories[0].history['accuracy']
val_loss = histories[0].history['val_loss']
val_accuracy = histories[0].history['val_accuracy']

# Plot Loss and Accuracy
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()