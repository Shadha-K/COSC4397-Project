MODE ?= naive  # default mode is naive

CC = g++
NVCC = nvcc
CFLAGS = -std=c++11 -O3 -Wall
NVCCFLAGS = -std=c++11 -O3 -arch=sm_60 # Adjust sm_60 to match your GPU architecture
CUDAFLAGS = -lcuda -lcudart -lcublas -lcurand

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Target executable name
TARGET = $(BIN_DIR)/cifar_cnn_$(MODE)

C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)

# Select CUDA source files based on MODE
ifeq ($(MODE), naive)
    CUDA_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/layer.cu
else ifeq ($(MODE), shmem)
    CUDA_SOURCES = $(SRC_DIR)/main_shmem.cu $(SRC_DIR)/layer_shmem.cu
else ifeq ($(MODE), regtil)
    CUDA_SOURCES = $(SRC_DIR)/main_regtil.cu $(SRC_DIR)/layer_regtil.cu
else
    $(error Unknown MODE '$(MODE)'. Valid options are 'naive', 'shmem', or 'regtil')
endif

HEADERS = $(wildcard $(SRC_DIR)/*.h)

C_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))
OBJECTS = $(C_OBJECTS) $(CPP_OBJECTS) $(CUDA_OBJECTS)

# Compile .c files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Ensure build and bin directories exist
$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(BIN_DIR))

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(CUDAFLAGS)

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Download CIFAR-10 dataset if not present
download_cifar:
	@echo "Checking for CIFAR-10 dataset..."
	@if [ ! -d "data/cifar-10-batches-bin" ]; then \
		echo "Downloading CIFAR-10 binary version..."; \
		mkdir -p data; \
		wget -O data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz; \
		tar -xzf data/cifar-10-binary.tar.gz -C data; \
		echo "CIFAR-10 dataset downloaded successfully."; \
	else \
		echo "CIFAR-10 dataset already exists."; \
	fi

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

# Clean all generated files
distclean: clean
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build the CIFAR-10 LeNet CNN program (default)"
	@echo "  download_cifar   - Download CIFAR-10 dataset"
	@echo "  clean            - Remove object files and executable"
	@echo "  distclean        - Remove all generated files and directories"
	@echo "  help             - Display this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make [target] MODE=<mode>"
	@echo ""
	@echo "Supported modes:"
	@echo "  MODE=naive       - Build the naive version (default)"
	@echo "  MODE=shmem       - Build the shared memory optimized version"
	@echo "  MODE=regtil      - Build the register tiling optimized version"
	@echo ""
	@echo "Example:"
	@echo "  make download_cifar"
	@echo "  make MODE=regtil"
	@echo "  ./$(TARGET)"

.PHONY: all clean distclean download_cifar help
