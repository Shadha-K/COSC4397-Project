#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

// ReLU activation function
__device__ float relu_function(float v)
{
	return (v > 0.0f) ? v : 0.0f;
}

// For the final output layer, we'll keep softmax or sigmoid for classification
__device__ float softmax_function(float v)
{
	return 1.0f / (1.0f + exp(-v));
}

__global__ void apply_relu_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = relu_function(input[idx]);
	}
}

// Keep the softmax function for the output layer
__global__ void apply_softmax_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = softmax_function(input[idx]);
	}
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}

// Optimized convolutional layer forward propagation with register tiling
__global__ void fp_preact_c1(float input[32][32][3], float preact[6][28][28], float weight[6][5][5][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int total_size = 6*28*28;
    
    // Each thread handles multiple output positions
    for (int n = pos; n < total_size; n += size) {
        const int f = n / (28*28);       // Filter number
        const int h = (n / 28) % 28;     // Output height
        const int w = n % 28;            // Output width
        
        // Register tiling - store filter weights in registers for this thread
        float reg_weight[5][5][3];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for (int c = 0; c < 3; c++) {
                    reg_weight[i][j][c] = weight[f][i][j][c];
                }
            }
        }
        
        // Register tiling - store input values needed for this output position
        float reg_input[5][5][3];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for (int c = 0; c < 3; c++) {
                    reg_input[i][j][c] = input[h + i][w + j][c];
                }
            }
        }
        
        float sum = 0.0f;
        
        // Compute convolution for this position using register values
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for (int c = 0; c < 3; c++) {
                    sum += reg_weight[i][j][c] * reg_input[i][j][c];
                }
            }
        }
        
        preact[f][h][w] = sum;
    }
}

__global__ void fp_bias_c1(float preact[6][28][28], float bias[6])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int total_size = 6*28*28;
    
    // Register tiling for bias
    float reg_bias[6];
    for (int f = 0; f < 6; f++) {
        reg_bias[f] = bias[f];
    }
    
    // Each thread handles multiple positions
    for (int n = pos; n < total_size; n += size) {
        const int f = n / (28*28);       // Filter number
        const int h = (n / 28) % 28;     // Output height
        const int w = n % 28;            // Output width
        
        preact[f][h][w] += reg_bias[f];
    }
}

// Optimized subsampling layer forward propagation with register tiling
__global__ void fp_preact_s1(float input[6][28][28], float preact[6][7][7], float weight[1][4][4])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int total_size = 6*7*7;
    
    // Register tiling for weights
    float reg_weight[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            reg_weight[i][j] = weight[0][i][j];
        }
    }
    
    // Each thread handles multiple output positions
    for (int n = pos; n < total_size; n += size) {
        const int c = n / (7*7);        // Channel
        const int h = (n / 7) % 7;      // Output height
        const int w = n % 7;            // Output width
        
        // Register tiling for input values
        float reg_input[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                reg_input[i][j] = input[c][h * 4 + i][w * 4 + j];
            }
        }
        
        float sum = 0.0f;
        
        // Compute pooling for this position using register values
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                sum += reg_weight[i][j] * reg_input[i][j];
            }
        }
        
        preact[c][h][w] = sum;
    }
}

__global__ void fp_bias_s1(float preact[6][7][7], float bias[1])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int total_size = 6*7*7;
    
    // Register tiling for bias
    float reg_bias = bias[0];
    
    // Each thread handles multiple positions
    for (int n = pos; n < total_size; n += size) {
        const int c = n / (7*7);        // Channel
        const int h = (n / 7) % 7;      // Output height
        const int w = n % 7;            // Output width
        
        preact[c][h][w] += reg_bias;
    }
}

// Optimized fully connected layer forward propagation with register tiling
__global__ void fp_preact_f(float input[6][7][7], float preact[10], float weight[10][6][7][7])
{
    // Each thread computes one output neuron
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_idx < 10) {
        // We don't need to tile the weights for this function since each thread processes
        // a single output neuron independently
        float sum = 0.0f;
        
        // For fully connected layer, we directly compute the weighted sum
        for (int c = 0; c < 6; c++) {
            for (int h = 0; h < 7; h++) {
                for (int w = 0; w < 7; w++) {
                    sum += weight[neuron_idx][c][h][w] * input[c][h][w];
                }
            }
        }
        
        preact[neuron_idx] = sum;
    }
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
    // Each thread handles one output neuron
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_idx < 10) {
        preact[neuron_idx] += bias[neuron_idx];
    }
}

// Fully connected layer backpropagation for weights - using register tiling
__global__ void bp_weight_f(float d_weight[10][6][7][7], float d_preact[10], float p_output[6][7][7])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int idx = bx * blockDim.x + tx;
    
    // Each thread loads its own d_preact values into registers for reuse
    float reg_d_preact[10];
    for (int i = 0; i < 10; i++) {
        reg_d_preact[i] = d_preact[i];
    }
    
    // Process multiple elements per thread for better occupancy
    for (int n = idx; n < 10*6*7*7; n += stride) {
        const int o = n % 10;                 // Output neuron
        const int c = (n / 10) % 6;           // Channel
        const int h = (n / (10*6)) % 7;       // Height
        const int w = (n / (10*6*7)) % 7;     // Width
        
        // Use the register-cached value instead of shared memory
        d_weight[o][c][h][w] = reg_d_preact[o] * p_output[c][h][w];
    }
}

// Fully connected layer backpropagation for bias - using register tiling
__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
    // Each thread handles one bias update
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < 10) {
        // Directly load to register rather than shared memory
        float reg_d_preact = d_preact[idx];
        bias[idx] += dt * reg_d_preact;
    }
}

// Subsampling layer backpropagation for output gradients - using register tiling
__global__ void bp_output_s1(float d_output[6][7][7], float n_weight[10][6][7][7], float nd_preact[10])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int idx = bx * blockDim.x + tx;
    
    // Cache nd_preact in registers
    float reg_nd_preact[10];
    for (int i = 0; i < 10; i++) {
        reg_nd_preact[i] = nd_preact[i];
    }
    
    // Process multiple elements per thread
    for (int n = idx; n < 6*7*7; n += stride) {
        const int c = n % 6;                // Channel
        const int h = (n / 6) % 7;          // Height
        const int w = (n / (6*7)) % 7;      // Width
        
        float sum = 0.0f;
        for (int o = 0; o < 10; o++) {
            sum += n_weight[o][c][h][w] * reg_nd_preact[o];
        }
        d_output[c][h][w] = sum;
    }
}

// Subsampling layer backpropagation for preactivation gradients - using register tiling
__global__ void bp_preact_s1(float d_preact[6][7][7], float d_output[6][7][7], float preact[6][7][7])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int idx = bx * blockDim.x + tx;
    
    // This kernel was already optimized for register usage
    for (int n = idx; n < 6*7*7; n += stride) {
        const int c = n % 6;                // Channel
        const int h = (n / 6) % 7;          // Height
        const int w = (n / (6*7)) % 7;      // Width
        
        // ReLU derivative is 1 if input > 0, else 0
        const float relu_deriv = preact[c][h][w] > 0.0f ? 1.0f : 0.0f;
        d_preact[c][h][w] = d_output[c][h][w] * relu_deriv;
    }
}

// Subsampling layer backpropagation for weights - using register tiling
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][7][7], float p_output[6][28][28])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int filter_idx = bx % 16; // 4x4 filter elements
    const int h = filter_idx / 4;
    const int w = filter_idx % 4;
    
    // Use register instead of shared memory for accumulation
    float reg_grad = 0.0f;
    
    // Accumulate gradients from all relevant positions
    for (int n = tx; n < 6*7*7; n += blockDim.x) {
        const int c = n % 6;                // Channel
        const int oh = (n / 6) % 7;         // Output height
        const int ow = (n / (6*7)) % 7;     // Output width
        
        reg_grad += d_preact[c][oh][ow] * p_output[c][oh * 4 + h][ow * 4 + w];
    }
    
    // Warp-level reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        reg_grad += __shfl_down_sync(0xffffffff, reg_grad, offset);
    }
    
    // Only the first thread in each warp updates the gradient
    if (tx % 32 == 0 && h < 4 && w < 4) {
        atomicAdd(&d_weight[0][h][w], reg_grad);
    }
}

// Subsampling layer backpropagation for bias - using register tiling
__global__ void bp_bias_s1(float bias[1], float d_preact[6][7][7])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int idx = bx * blockDim.x + tx;
    
    // Use register instead of shared memory
    float reg_grad_sum = 0.0f;
    
    // Each thread accumulates gradients
    for (int n = idx; n < 6*7*7; n += blockDim.x * gridDim.x) {
        const int c = n % 6;                // Channel
        const int h = (n / 6) % 7;          // Height
        const int w = (n / (6*7)) % 7;      // Width
        
        reg_grad_sum += d_preact[c][h][w];
    }
    
    // Warp-level reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        reg_grad_sum += __shfl_down_sync(0xffffffff, reg_grad_sum, offset);
    }
    
    // Only the first thread in each warp contributes to the atomic add
    if (tx % 32 == 0) {
        atomicAdd(&bias[0], dt * reg_grad_sum / (6.0f * 7.0f * 7.0f));
    }
}

// Convolutional layer backpropagation for output gradients - using register tiling
__global__ void bp_output_c1(float d_output[6][28][28], float n_weight[1][4][4], float nd_preact[6][7][7])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int idx = bx * blockDim.x + tx;
    
    // Cache weights in registers
    float reg_weight[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            reg_weight[i][j] = n_weight[0][i][j];
        }
    }
    
    // Process multiple elements per thread
    for (int n = idx; n < 6*28*28; n += blockDim.x * gridDim.x) {
        const int c = n % 6;                  // Channel
        const int h = (n / 6) % 28;           // Height
        const int w = (n / (6*28)) % 28;      // Width
        
        // Find corresponding position in nd_preact
        const int oh = h / 4;
        const int ow = w / 4;
        const int kh = h % 4;
        const int kw = w % 4;
        
        // Only update if we're within bounds and if there's a corresponding preact gradient
        if (oh < 7 && ow < 7) {
            d_output[c][h][w] = reg_weight[kh][kw] * nd_preact[c][oh][ow];
        }
    }
}

// Convolutional layer backpropagation for preactivation gradients - using register tiling
__global__ void bp_preact_c1(float d_preact[6][28][28], float d_output[6][28][28], float preact[6][28][28])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int idx = bx * blockDim.x + tx;
    
    // This kernel was already optimized for register usage
    for (int n = idx; n < 6*28*28; n += stride) {
        const int c = n % 6;                  // Channel
        const int h = (n / 6) % 28;           // Height
        const int w = (n / (6*28)) % 28;      // Width
        
        // ReLU derivative
        const float relu_deriv = preact[c][h][w] > 0.0f ? 1.0f : 0.0f;
        d_preact[c][h][w] = d_output[c][h][w] * relu_deriv;
    }
}

// Convolutional layer backpropagation for weights - using register tiling
__global__ void bp_weight_c1(float d_weight[6][5][5][3], float d_preact[6][28][28], float p_output[32][32][3])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    
    // Each block handles one filter element for one channel
    const int filter = bx / (5*5*3);       // Filter index (0-5)
    const int remaining = bx % (5*5*3);
    const int kh = remaining / (5*3);      // Kernel height
    const int remaining2 = remaining % (5*3);
    const int kw = remaining2 / 3;         // Kernel width
    const int kc = remaining2 % 3;         // Kernel channel
    
    // Use register instead of shared memory
    float reg_grad = 0.0f;
    
    // Each thread computes gradient for a subset of output positions
    for (int n = tx; n < 28*28; n += blockDim.x) {
        const int oh = n / 28;             // Output height
        const int ow = n % 28;             // Output width
        
        if (filter < 6 && kh < 5 && kw < 5 && kc < 3 && oh < 28 && ow < 28) {
            reg_grad += d_preact[filter][oh][ow] * p_output[oh + kh][ow + kw][kc];
        }
    }
    
    // Warp-level reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        reg_grad += __shfl_down_sync(0xffffffff, reg_grad, offset);
    }
    
    // Only the first thread in each warp updates the weight gradient
    if (tx % 32 == 0 && filter < 6 && kh < 5 && kw < 5 && kc < 3) {
        atomicAdd(&d_weight[filter][kh][kw][kc], reg_grad / (28.0f * 28.0f));
    }
}

// Convolutional layer backpropagation for bias - using register tiling
__global__ void bp_bias_c1(float bias[6], float d_preact[6][28][28])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int filter = bx;  // Each block handles one filter
    
    // Use register instead of shared memory
    float reg_grad = 0.0f;
    
    // Accumulate gradients for this filter
    if (filter < 6) {
        for (int n = tx; n < 28*28; n += blockDim.x) {
            const int h = n / 28;
            const int w = n % 28;
            reg_grad += d_preact[filter][h][w];
        }
    }
    
    // Warp-level reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        reg_grad += __shfl_down_sync(0xffffffff, reg_grad, offset);
    }
    
    // Only the first thread in each warp updates the bias
    if (tx % 32 == 0 && filter < 6) {
        atomicAdd(&bias[filter], dt * reg_grad / (28.0f * 28.0f));
    }
}