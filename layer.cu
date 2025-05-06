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

// Updated for CIFAR-10: 32x32x3 input, 6 filters of size 5x5x3
__global__ void fp_preact_c1(float input[32][32][3], float preact[6][28][28], float weight[6][5][5][3])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 5*5*3*6*28*28;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 5);    // Filter height dimension
		const int i2 = ((idx /= 5	) % 5);    // Filter width dimension
		const int i3 = ((idx /= 5	) % 3);    // Filter channel dimension
		const int i4 = ((idx /= 3	) % 6);    // Filter number
		const int i5 = ((idx /= 6	) % 28);   // Output height
		const int i6 = ((idx /= 28	) % 28);   // Output width

		atomicAdd(&preact[i4][i5][i6], weight[i4][i1][i2][i3] * input[i5 + i1][i6 + i2][i3]);
	}
}

__global__ void fp_bias_c1(float preact[6][28][28], float bias[6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*28*28;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);    // Filter number
		const int i2 = ((idx /= 6	) % 28);   // Output height
		const int i3 = ((idx /= 28	) % 28);   // Output width

		preact[i1][i2][i3] += bias[i1];
	}
}

// Updated for CIFAR-10: 6x28x28 input, 6x7x7 output, 4x4 pooling
__global__ void fp_preact_s1(float input[6][28][28], float preact[6][7][7], float weight[1][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4*6*7*7;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);    // Pooling height
		const int i2 = ((idx /= 4	) % 4);    // Pooling width
		const int i3 = ((idx /= 4	) % 6);    // Channel
		const int i4 = ((idx /= 6	) % 7);    // Output height
		const int i5 = ((idx /= 7	) % 7);    // Output width

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

__global__ void fp_bias_s1(float preact[6][7][7], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*7*7;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);    // Channel
		const int i2 = ((idx /= 6	) % 7);    // Output height
		const int i3 = ((idx /= 7	) % 7);    // Output width

		preact[i1][i2][i3] += bias[0];
	}
}

// Updated for CIFAR-10: 6x7x7 input, 10 output neurons
__global__ void fp_preact_f(float input[6][7][7], float preact[10], float weight[10][6][7][7])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*7*7;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);   // Output neuron
		const int i2 = ((idx /= 10	) % 6);    // Channel
		const int i3 = ((idx /= 6	) % 7);    // Input height
		const int i4 = ((idx /= 7	) % 7);    // Input width

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

// Updated for CIFAR-10: 10 output neurons, 6x7x7 input
__global__ void bp_weight_f(float d_weight[10][6][7][7], float d_preact[10], float p_output[6][7][7])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*7*7;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);   // Output neuron
		const int i2 = ((idx /= 10	) % 6);    // Channel
		const int i3 = ((idx /= 6	) % 7);    // Input height
		const int i4 = ((idx /= 7	) % 7);    // Input width

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

// Updated for CIFAR-10: 6x7x7 output error, 10x6x7x7 weights
__global__ void bp_output_s1(float d_output[6][7][7], float n_weight[10][6][7][7], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*7*7;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);   // Output neuron
		const int i2 = ((idx /= 10	) % 6);    // Channel
		const int i3 = ((idx /= 6	) % 7);    // Input height
		const int i4 = ((idx /= 7	) % 7);    // Input width

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

// Updated for CIFAR-10: 6x7x7 dimensions, using ReLU derivative
__global__ void bp_preact_s1(float d_preact[6][7][7], float d_output[6][7][7], float preact[6][7][7])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*7*7;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);    // Channel
		const int i2 = ((idx /= 6	) % 7);    // Height
		const int i3 = ((idx /= 7	) % 7);    // Width

		// ReLU derivative is 1 if input > 0, else 0
		const float relu_deriv = preact[i1][i2][i3] > 0.0f ? 1.0f : 0.0f;

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * relu_deriv;
	}
}

// Updated for CIFAR-10: 1x4x4 weight gradients, 6x7x7 preact gradients, 6x28x28 output
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][7][7], float p_output[6][28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*7*7;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);    // Single weight filter
		const int i2 = ((idx /= 1	) % 4);    // Filter height
		const int i3 = ((idx /= 4	) % 4);    // Filter width
		const int i4 = ((idx /= 4	) % 6);    // Channel
		const int i5 = ((idx /= 6	) % 7);    // Output height
		const int i6 = ((idx /= 7	) % 7);    // Output width

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

// Updated for CIFAR-10: 6x7x7 preact gradients
__global__ void bp_bias_s1(float bias[1], float d_preact[6][7][7])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*7*7;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);    // Channel
		const int i2 = ((idx /= 6	) % 7);    // Height
		const int i3 = ((idx /= 7	) % 7);    // Width

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
	}
}

// Updated for CIFAR-10: 6x28x28 output error, 1x4x4 weights, 6x7x7 preact gradients
__global__ void bp_output_c1(float d_output[6][28][28], float n_weight[1][4][4], float nd_preact[6][7][7])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*7*7;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);    // Single weight filter
		const int i2 = ((idx /= 1	) % 4);    // Filter height
		const int i3 = ((idx /= 4	) % 4);    // Filter width
		const int i4 = ((idx /= 4	) % 6);    // Channel
		const int i5 = ((idx /= 6	) % 7);    // Output gradient height
		const int i6 = ((idx /= 7	) % 7);    // Output gradient width

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

// Updated for CIFAR-10: 6x28x28 dimensions, using ReLU derivative
__global__ void bp_preact_c1(float d_preact[6][28][28], float d_output[6][28][28], float preact[6][28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*28*28;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);    // Channel
		const int i2 = ((idx /= 6	) % 28);   // Height
		const int i3 = ((idx /= 28	) % 28);   // Width

		// ReLU derivative is 1 if input > 0, else 0
		const float relu_deriv = preact[i1][i2][i3] > 0.0f ? 1.0f : 0.0f;

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * relu_deriv;
	}
}

// Updated for CIFAR-10: 6x5x5x3 weight gradients, 6x28x28 preact gradients, 32x32x3 input
__global__ void bp_weight_c1(float d_weight[6][5][5][3], float d_preact[6][28][28], float p_output[32][32][3])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*5*5*3*28*28;
	const float d = pow(28.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);    // Filter number
		const int i2 = ((idx /= 6	) % 5);    // Filter height
		const int i3 = ((idx /= 5	) % 5);    // Filter width
		const int i4 = ((idx /= 5	) % 3);    // Filter channel
		const int i5 = ((idx /= 3	) % 28);   // Output height
		const int i6 = ((idx /= 28	) % 28);   // Output width

		atomicAdd(&d_weight[i1][i2][i3][i4], d_preact[i1][i5][i6] * p_output[i5 + i2][i6 + i3][i4] / d);
	}
}

// Updated for CIFAR-10: 6x28x28 preact gradients
__global__ void bp_bias_c1(float bias[6], float d_preact[6][28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*28*28;
	const float d = pow(28.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);    // Filter number
		const int i2 = ((idx /= 6	) % 28);   // Height
		const int i3 = ((idx /= 28	) % 28);   // Width

		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
	}
}