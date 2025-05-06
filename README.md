# CIFAR-10 CNN Classifier

A CUDA-based Convolutional Neural Network implementation for CIFAR-10 image classification with multiple optimization strategies.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (compatible with architecture sm_60 or adjust in Makefile)
- g++ compiler
- wget (for dataset download)

## Quick Start

```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Download the CIFAR-10 dataset
make download_cifar

# Build with default optimization (naive implementation)
make

# Run the program
./bin/cifar_cnn_naive
```

## Optimization Modes

This implementation supports three optimization modes:

1. **Naive** (default): Basic CUDA implementation
   ```bash
   make MODE=naive
   ./bin/cifar_cnn_naive
   ```

2. **Shared Memory** optimization:
   ```bash
   make MODE=shmem
   ./bin/cifar_cnn_shmem
   ```

3. **Register Tiling** optimization:
   ```bash
   make MODE=regtil
   ./bin/cifar_cnn_regtil
   ```

## Cleaning Up

- Remove object files and executable:
  ```bash
  make clean
  ```

- Remove all generated files:
  ```bash
  make distclean
  ```

## GPU Architecture

By default, the code compiles for compute capability 6.0 (sm_60). If your GPU has a different architecture, modify the `NVCCFLAGS` in the Makefile:

```makefile
NVCCFLAGS = -std=c++11 -O3 -arch=sm_XX # Replace XX with your architecture version
```

## Project Structure

- `main.cu`/`main_*.cu`: Main program entry points for different optimization modes
- `layer.cu`/`layer_*.cu`: CNN layer implementations
- `*.h`: Header files with declarations and structures
- `data/`: Directory containing the CIFAR-10 dataset after download
