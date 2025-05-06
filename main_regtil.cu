#include "cifar_loader.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>
#include <cublas_v2.h> // Add CUBLAS header
#include <curand.h>
#include <math.h>

// Define layers of CNN
// Input: 32x32x3 (RGB image)
// C1: Convolutional layer with 6 feature maps of size 28x28
// S1: Subsampling layer with 6 feature maps of size 7x7
// F: Fully connected layer with 10 outputs (10 classes in CIFAR-10)
static Layer l_input = Layer(0, 0, 32*32*3);
static Layer l_c1 = Layer(5*5*3, 6, 28*28*6); // 5x5 kernel, 6 features, 3 input channels
static Layer l_s1 = Layer(4*4, 1, 7*7*6);     // 4x4 pooling, 6 channels
static Layer l_f = Layer(7*7*6, 10, 10);      // Fully connected layer

// Function declarations
static void initialize_weights();
static void learn();
static unsigned int classify(unsigned char data[32][32][3]);
static void test();
static double forward_pass(unsigned char data[32][32][3]);
static double back_pass();

// Threshold for learning
static const float threshold = 0.01f;

// Load CIFAR-10 data
static void loaddata()
{
    // Use the function from cifar_loader.c
    load_cifar_data("data/cifar-10-batches-bin");
}

// He initialization for weights (sqrt(2/n_in))
static void initialize_weights()
{
    // Create CURAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    // Set seed
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    
    // He initialization factors for each layer
    // C1 layer: input size is 5*5*3 = 75
    float c1_factor = sqrt(2.0f / (5*5*3));
    
    // S1 layer: input size is 4*4 = 16
    float s1_factor = sqrt(2.0f / (4*4));
    
    // F layer: input size is 7*7*6 = 294
    float f_factor = sqrt(2.0f / (7*7*6));
    
    // Generate random numbers for weights on device
    curandGenerateNormal(gen, l_c1.weight, l_c1.M * l_c1.N, 0.0f, c1_factor);
    curandGenerateNormal(gen, l_s1.weight, l_s1.M * l_s1.N, 0.0f, s1_factor);
    curandGenerateNormal(gen, l_f.weight, l_f.M * l_f.N, 0.0f, f_factor);
    
    // Initialize biases to zero
    cudaMemset(l_c1.bias, 0, sizeof(float) * l_c1.N);
    cudaMemset(l_s1.bias, 0, sizeof(float) * l_s1.N);
    cudaMemset(l_f.bias, 0, sizeof(float) * l_f.N);
    
    // Destroy generator
    curandDestroyGenerator(gen);
    
    printf("Weights initialized with He initialization\n");
}

int main(int argc, const char **argv)
{
    srand(time(NULL));

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialization failed with error code - %d\n", err);
        return 1;
    }

    loaddata();
    initialize_weights(); // Initialize weights with He initialization
    learn();
    test();

    return 0;
}

// Forward propagation of a single image in dataset
static double forward_pass(unsigned char data[32][32][3])
{
    float input[32][32][3];

    // Convert unsigned char to float and normalize to [0,1] range
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                input[i][j][c] = data[i][j][c] / 255.0f;
            }
        }
    }

    l_input.clear();
    l_c1.clear();
    l_s1.clear();
    l_f.clear();

    clock_t start, end;
    start = clock();

    l_input.setOutput((float *)input);
    
    // Forward pass through convolutional layer C1
    fp_preact_c1<<<64, 64>>>((float (*)[32][3])l_input.output, (float (*)[28][28])l_c1.preact, (float (*)[5][5][3])l_c1.weight);
    fp_bias_c1<<<64, 64>>>((float (*)[28][28])l_c1.preact, l_c1.bias);
    // Using ReLU activation instead of step function
    apply_relu_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

    // Forward pass through subsampling layer S1
    fp_preact_s1<<<64, 64>>>((float (*)[28][28])l_c1.output, (float (*)[7][7])l_s1.preact, (float (*)[4][4])l_s1.weight);
    fp_bias_s1<<<64, 64>>>((float (*)[7][7])l_s1.preact, l_s1.bias);
    // Using ReLU activation instead of step function
    apply_relu_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);

    // Forward pass through fully connected layer F
    fp_preact_f<<<1, 32>>>((float (*)[7][7])l_s1.output, l_f.preact, (float (*)[6][7][7])l_f.weight);
    fp_bias_f<<<1, 32>>>(l_f.preact, l_f.bias);
    // Using softmax activation for output layer instead of step function
    apply_softmax_function<<<1, 32>>>(l_f.preact, l_f.output, l_f.O);
    
    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
    clock_t start, end;
    start = clock();

    // Grid and block dimensions optimized for register tiling
    dim3 gridDim(64);
    dim3 blockDim(256);  // Increased thread count for better occupancy with register tiling
    
    // Block size for bias kernels (smaller since each block processes fewer elements)
    dim3 biasDim(128);

    // Backpropagate through fully connected layer
    bp_weight_f<<<gridDim, blockDim>>>((float (*)[6][7][7])l_f.d_weight, l_f.d_preact, (float (*)[7][7])l_s1.output);
    bp_bias_f<<<1, 32>>>(l_f.bias, l_f.d_preact);  // Only need 10 threads, so using 32 is sufficient

    // Backpropagate through subsampling layer
    bp_output_s1<<<gridDim, blockDim>>>((float (*)[7][7])l_s1.d_output, (float (*)[6][7][7])l_f.weight, l_f.d_preact);
    bp_preact_s1<<<gridDim, blockDim>>>((float (*)[7][7])l_s1.d_preact, (float (*)[7][7])l_s1.d_output, (float (*)[7][7])l_s1.preact);
    bp_weight_s1<<<16, 256>>>((float (*)[4][4])l_s1.d_weight, (float (*)[7][7])l_s1.d_preact, (float (*)[28][28])l_c1.output);
    bp_bias_s1<<<1, 256>>>(l_s1.bias, (float (*)[7][7])l_s1.d_preact);

    // Backpropagate through convolutional layer
    bp_output_c1<<<gridDim, blockDim>>>((float (*)[28][28])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[7][7])l_s1.d_preact);
    bp_preact_c1<<<gridDim, blockDim>>>((float (*)[28][28])l_c1.d_preact, (float (*)[28][28])l_c1.d_output, (float (*)[28][28])l_c1.preact);
    
    // For weight_c1, we need more blocks to handle all filter elements
    bp_weight_c1<<<6*5*5*3, 256>>>((float (*)[5][5][3])l_c1.d_weight, (float (*)[28][28])l_c1.d_preact, (float (*)[32][3])l_input.output);
    
    // For bias_c1, we need exactly 6 blocks (one per filter)
    bp_bias_c1<<<6, 256>>>(l_c1.bias, (float (*)[28][28])l_c1.d_preact);

    // Apply gradients
    apply_grad<<<64, 256>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
    apply_grad<<<64, 256>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
    apply_grad<<<64, 256>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);

    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static void learn()
{
    cublasHandle_t blas;
    cublasCreate(&blas);

    float err;
    const int max_iter = 20;
    int iter = max_iter;
    
    double time_taken = 0.0;

    fprintf(stdout, "Training LeNet on CIFAR-10 using ReLU activation...\n");

    while (iter < 0 || iter-- > 0) {
        err = 0.0f;

        for (int i = 0; i < train_cnt; ++i) {
            float tmp_err;

            // Convert train_set[i].data from float to unsigned char for forward_pass
            unsigned char temp_data[32][32][3];
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < 32; ++h) {
                    for (int w = 0; w < 32; ++w) {
                        temp_data[h][w][c] = (unsigned char)(train_set[i].data[h][w][c] * 255.0f);
                    }
                }
            }

            time_taken += forward_pass(temp_data);

            l_f.bp_clear();
            l_s1.bp_clear();
            l_c1.bp_clear();

            // Cross-entropy loss for softmax output
            makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
            cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
            err += tmp_err;

            time_taken += back_pass();
            
            // Print progress periodically
            if (i % 1000 == 0) {
                fprintf(stdout, "Training sample %d/%d\r", i, train_cnt);
                fflush(stdout);
            }
        }

        err /= train_cnt;
        fprintf(stdout, "Epoch %d complete, error: %e, time_on_gpu: %lf\n", max_iter - iter, err, time_taken);

        if (err < threshold) {
            fprintf(stdout, "Training complete, error less than threshold\n\n");
            break;
        }
    }
    
    fprintf(stdout, "\nTotal training time: %lf seconds\n", time_taken);
    
    cublasDestroy(blas);
}

// Returns label of given data (0-9)
static unsigned int classify(unsigned char data[32][32][3])
{
    float res[10];

    forward_pass(data);

    unsigned int max = 0;

    cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

    for (int i = 1; i < 10; ++i) {
        if (res[max] < res[i]) {
            max = i;
        }
    }

    return max;
}

// Perform forward propagation of test data
static void test()
{
    int error = 0;
    int confusion_matrix[10][10] = {0};

    fprintf(stdout, "Testing model on %d samples...\n", test_cnt);

    for (int i = 0; i < test_cnt; ++i) {
        // Convert test_set[i].data from float to unsigned char for classify
        unsigned char temp_data[32][32][3];
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 32; ++h) {
                for (int w = 0; w < 32; ++w) {
                    temp_data[h][w][c] = (unsigned char)(test_set[i].data[h][w][c] * 255.0f);
                }
            }
        }

        unsigned int predicted = classify(temp_data);
        unsigned int actual = test_set[i].label;
        
        confusion_matrix[actual][predicted]++;
        
        if (predicted != actual) {
            ++error;
        }
        
        // Print progress periodically
        if (i % 100 == 0) {
            fprintf(stdout, "Testing sample %d/%d\r", i, test_cnt);
            fflush(stdout);
        }
    }

    float error_rate = (float)error / (float)test_cnt * 100.0f;
    float accuracy = 100.0f - error_rate;
    
    fprintf(stdout, "\nTest Results:\n");
    fprintf(stdout, "Errors: %d/%d\n", error, test_cnt);
    fprintf(stdout, "Accuracy: %.2f%%\n", accuracy);
    fprintf(stdout, "Error Rate: %.2f%%\n", error_rate);
    
    // Print confusion matrix
    fprintf(stdout, "\nConfusion Matrix:\n");
    fprintf(stdout, "    | ");
    for (int i = 0; i < 10; i++) {
        fprintf(stdout, "%4d ", i);
    }
    fprintf(stdout, "\n----+-");
    for (int i = 0; i < 10; i++) {
        fprintf(stdout, "-----");
    }
    fprintf(stdout, "\n");
    
    for (int i = 0; i < 10; i++) {
        fprintf(stdout, "%3d | ", i);
        for (int j = 0; j < 10; j++) {
            fprintf(stdout, "%4d ", confusion_matrix[i][j]);
        }
        fprintf(stdout, "\n");
    }
}