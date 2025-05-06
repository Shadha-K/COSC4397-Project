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

// Optimized convolutional layer forward propagation
__global__ void fp_preact_c1(float input[32][32][3], float preact[6][28][28], float weight[6][5][5][3])
{
    // Use shared memory for the weights
    __shared__ float shared_weight[6][5][5][3];
    
    // Load weights into shared memory
    const int tx = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Each thread loads multiple weight values
    for (int i = tx; i < 6*5*5*3; i += num_threads) {
        const int f = i / (5*5*3);           // Filter number
        const int remainder = i % (5*5*3);
        const int h = remainder / (5*3);     // Filter height
        const int remainder2 = remainder % (5*3);
        const int w = remainder2 / 3;        // Filter width
        const int c = remainder2 % 3;        // Channel
        
        if (f < 6 && h < 5 && w < 5 && c < 3) {
            shared_weight[f][h][w][c] = weight[f][h][w][c];
        }
    }
    __syncthreads();
    
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int total_size = 6*28*28;
    
    // Each thread handles multiple output positions
    for (int n = pos; n < total_size; n += size) {
        const int f = n / (28*28);       // Filter number
        const int h = (n / 28) % 28;     // Output height
        const int w = n % 28;            // Output width
        
        float sum = 0.0f;
        
        // Compute convolution for this position
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for (int c = 0; c < 3; c++) {
                    sum += shared_weight[f][i][j][c] * input[h + i][w + j][c];
                }
            }
        }
        
        preact[f][h][w] = sum;
    }
}

__global__ void fp_bias_c1(float preact[6][28][28], float bias[6])
{
    // Use shared memory for bias
    __shared__ float shared_bias[6];
    
    // Load bias into shared memory
    if (threadIdx.x < 6) {
        shared_bias[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();
    
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int total_size = 6*28*28;
    
    // Each thread handles multiple positions
    for (int n = pos; n < total_size; n += size) {
        const int f = n / (28*28);       // Filter number
        const int h = (n / 28) % 28;     // Output height
        const int w = n % 28;            // Output width
        
        preact[f][h][w] += shared_bias[f];
    }
}

// Optimized subsampling layer forward propagation
__global__ void fp_preact_s1(float input[6][28][28], float preact[6][7][7], float weight[1][4][4])
{
    // Use shared memory for the weights
    __shared__ float shared_weight[4][4];
    
    // Load weights into shared memory
    if (threadIdx.x < 16) {
        const int i = threadIdx.x / 4;
        const int j = threadIdx.x % 4;
        shared_weight[i][j] = weight[0][i][j];
    }
    __syncthreads();
    
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int total_size = 6*7*7;
    
    // Each thread handles multiple output positions
    for (int n = pos; n < total_size; n += size) {
        const int c = n / (7*7);        // Channel
        const int h = (n / 7) % 7;      // Output height
        const int w = n % 7;            // Output width
        
        float sum = 0.0f;
        
        // Compute pooling for this position
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                sum += shared_weight[i][j] * input[c][h * 4 + i][w * 4 + j];
            }
        }
        
        preact[c][h][w] = sum;
    }
}

__global__ void fp_bias_s1(float preact[6][7][7], float bias[1])
{
    // Use shared memory for bias
    __shared__ float shared_bias;
    
    // Load bias into shared memory
    if (threadIdx.x == 0) {
        shared_bias = bias[0];
    }
    __syncthreads();
    
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int total_size = 6*7*7;
    
    // Each thread handles multiple positions
    for (int n = pos; n < total_size; n += size) {
        const int c = n / (7*7);        // Channel
        const int h = (n / 7) % 7;      // Output height
        const int w = n % 7;            // Output width
        
        preact[c][h][w] += shared_bias;
    }
}

// Optimized fully connected layer forward propagation
__global__ void fp_preact_f(float input[6][7][7], float preact[10], float weight[10][6][7][7])
{
    // Each thread computes one output neuron
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_idx < 10) {
        float sum = 0.0f;
        
        // Sum over all input values
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

// Optimized kernels with shared memory for CIFAR-10

// Fully connected layer backpropagation for weights
__global__ void bp_weight_f(float d_weight[10][6][7][7], float d_preact[10], float p_output[6][7][7])
{
    __shared__ float shared_d_preact[10];
    
    // Load d_preact into shared memory
    if (threadIdx.x < 10) {
        shared_d_preact[threadIdx.x] = d_preact[threadIdx.x];
    }
    __syncthreads();
    
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int idx = bx * blockDim.x + tx;
    
    // Process multiple elements per thread for better occupancy
    for (int n = idx; n < 10*6*7*7; n += stride) {
        const int o = n % 10;                 // Output neuron
        const int c = (n / 10) % 6;           // Channel
        const int h = (n / (10*6)) % 7;       // Height
        const int w = (n / (10*6*7)) % 7;     // Width
        
        d_weight[o][c][h][w] = shared_d_preact[o] * p_output[c][h][w];
    }
}

// Fully connected layer backpropagation for bias
__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
    __shared__ float shared_d_preact[10];
    
    // Load d_preact into shared memory
    if (threadIdx.x < 10) {
        shared_d_preact[threadIdx.x] = d_preact[threadIdx.x];
    }
    __syncthreads();
    
    // Each thread handles one bias update
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 10) {
        bias[idx] += dt * shared_d_preact[idx];
    }
}

// Subsampling layer backpropagation for output gradients
__global__ void bp_output_s1(float d_output[6][7][7], float n_weight[10][6][7][7], float nd_preact[10])
{
    __shared__ float shared_nd_preact[10];
    
    // Load nd_preact into shared memory
    if (threadIdx.x < 10) {
        shared_nd_preact[threadIdx.x] = nd_preact[threadIdx.x];
    }
    __syncthreads();
    
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int idx = bx * blockDim.x + tx;
    
    // Process multiple elements per thread
    for (int n = idx; n < 6*7*7; n += stride) {
        const int c = n % 6;                // Channel
        const int h = (n / 6) % 7;          // Height
        const int w = (n / (6*7)) % 7;      // Width
        
        float sum = 0.0f;
        for (int o = 0; o < 10; o++) {
            sum += n_weight[o][c][h][w] * shared_nd_preact[o];
        }
        d_output[c][h][w] = sum;
    }
}

// Subsampling layer backpropagation for preactivation gradients
__global__ void bp_preact_s1(float d_preact[6][7][7], float d_output[6][7][7], float preact[6][7][7])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int idx = bx * blockDim.x + tx;
    
    for (int n = idx; n < 6*7*7; n += stride) {
        const int c = n % 6;                // Channel
        const int h = (n / 6) % 7;          // Height
        const int w = (n / (6*7)) % 7;      // Width
        
        // ReLU derivative is 1 if input > 0, else 0
        const float relu_deriv = preact[c][h][w] > 0.0f ? 1.0f : 0.0f;
        d_preact[c][h][w] = d_output[c][h][w] * relu_deriv;
    }
}

// Subsampling layer backpropagation for weights
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][7][7], float p_output[6][28][28])
{
    __shared__ float shared_grad[32]; // Temporary storage for partial sums
    
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int filter_idx = bx % 16; // 4x4 filter elements
    const int h = filter_idx / 4;
    const int w = filter_idx % 4;
    
    // Initialize shared memory
    shared_grad[tx] = 0.0f;
    
    // Accumulate gradients from all relevant positions
    for (int n = tx; n < 6*7*7; n += blockDim.x) {
        const int c = n % 6;                // Channel
        const int oh = (n / 6) % 7;         // Output height
        const int ow = (n / (6*7)) % 7;     // Output width
        
        shared_grad[tx] += d_preact[c][oh][ow] * p_output[c][oh * 4 + h][ow * 4 + w];
    }
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            shared_grad[tx] += shared_grad[tx + stride];
        }
        __syncthreads();
    }
    
    // Update weight gradient
    if (tx == 0 && h < 4 && w < 4) {
        atomicAdd(&d_weight[0][h][w], shared_grad[0]);
    }
}

// Subsampling layer backpropagation for bias
__global__ void bp_bias_s1(float bias[1], float d_preact[6][7][7])
{
    __shared__ float grad_sum[256];
    
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int idx = bx * blockDim.x + tx;
    
    // Initialize shared memory
    grad_sum[tx] = 0.0f;
    
    // Each thread accumulates gradients
    for (int n = idx; n < 6*7*7; n += blockDim.x * gridDim.x) {
        const int c = n % 6;                // Channel
        const int h = (n / 6) % 7;          // Height
        const int w = (n / (6*7)) % 7;      // Width
        
        grad_sum[tx] += d_preact[c][h][w];
    }
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            grad_sum[tx] += grad_sum[tx + stride];
        }
        __syncthreads();
    }
    
    // Only one thread updates the bias
    if (tx == 0) {
        atomicAdd(&bias[0], dt * grad_sum[0] / (6.0f * 7.0f * 7.0f));
    }
}

// Convolutional layer backpropagation for output gradients
__global__ void bp_output_c1(float d_output[6][28][28], float n_weight[1][4][4], float nd_preact[6][7][7])
{
    __shared__ float shared_weight[4][4];
    
    // Load weights into shared memory
    if (threadIdx.x < 16) {
        const int h = threadIdx.x / 4;
        const int w = threadIdx.x % 4;
        shared_weight[h][w] = n_weight[0][h][w];
    }
    __syncthreads();
    
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int idx = bx * blockDim.x + tx;
    
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
            d_output[c][h][w] = shared_weight[kh][kw] * nd_preact[c][oh][ow];
        }
    }
}

// Convolutional layer backpropagation for preactivation gradients
__global__ void bp_preact_c1(float d_preact[6][28][28], float d_output[6][28][28], float preact[6][28][28])
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int idx = bx * blockDim.x + tx;
    
    for (int n = idx; n < 6*28*28; n += stride) {
        const int c = n % 6;                  // Channel
        const int h = (n / 6) % 28;           // Height
        const int w = (n / (6*28)) % 28;      // Width
        
        // ReLU derivative
        const float relu_deriv = preact[c][h][w] > 0.0f ? 1.0f : 0.0f;
        d_preact[c][h][w] = d_output[c][h][w] * relu_deriv;
    }
}

// Convolutional layer backpropagation for weights
__global__ void bp_weight_c1(float d_weight[6][5][5][3], float d_preact[6][28][28], float p_output[32][32][3])
{
    __shared__ float shared_grad[32];
    
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    
    // Each block handles one filter element for one channel
    const int filter = bx / (5*5*3);       // Filter index (0-5)
    const int remaining = bx % (5*5*3);
    const int kh = remaining / (5*3);      // Kernel height
    const int remaining2 = remaining % (5*3);
    const int kw = remaining2 / 3;         // Kernel width
    const int kc = remaining2 % 3;         // Kernel channel
    
    // Initialize shared memory
    shared_grad[tx] = 0.0f;
    
    // Each thread computes gradient for a subset of output positions
    for (int n = tx; n < 28*28; n += blockDim.x) {
        const int oh = n / 28;             // Output height
        const int ow = n % 28;             // Output width
        
        if (filter < 6 && kh < 5 && kw < 5 && kc < 3 && oh < 28 && ow < 28) {
            shared_grad[tx] += d_preact[filter][oh][ow] * p_output[oh + kh][ow + kw][kc];
        }
    }
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            shared_grad[tx] += shared_grad[tx + stride];
        }
        __syncthreads();
    }
    
    // Only one thread updates the weight gradient
    if (tx == 0 && filter < 6 && kh < 5 && kw < 5 && kc < 3) {
        atomicAdd(&d_weight[filter][kh][kw][kc], shared_grad[0] / (28.0f * 28.0f));
    }
}

// Convolutional layer backpropagation for bias
__global__ void bp_bias_c1(float bias[6], float d_preact[6][28][28])
{
    __shared__ float shared_grad[256];
    
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int filter = bx;  // Each block handles one filter
    
    // Initialize shared memory
    shared_grad[tx] = 0.0f;
    
    // Accumulate gradients for this filter
    if (filter < 6) {
        for (int n = tx; n < 28*28; n += blockDim.x) {
            const int h = n / 28;
            const int w = n % 28;
            shared_grad[tx] += d_preact[filter][h][w];
        }
    }
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            shared_grad[tx] += shared_grad[tx + stride];
        }
        __syncthreads();
    }
    
    // Update bias
    if (tx == 0 && filter < 6) {
        atomicAdd(&bias[filter], dt * shared_grad[0] / (28.0f * 28.0f));
    }
}