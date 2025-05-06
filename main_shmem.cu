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
    
    // Calculate optimal thread and block configuration for each layer
    // For convolutional layer C1 (6*28*28 = 4,704 elements)
    dim3 c1_block(256);  // 256 threads per block for better occupancy
    dim3 c1_grid((6*28*28 + c1_block.x - 1) / c1_block.x);  // Ceil(4704/256) = 19 blocks
    
    // For subsampling layer S1 (6*7*7 = 294 elements)
    dim3 s1_block(256);
    dim3 s1_grid((6*7*7 + s1_block.x - 1) / s1_block.x);  // Ceil(294/256) = 2 blocks
    
    // For fully connected layer F (10 elements)
    dim3 f_block(10);  // Only 10 output neurons
    dim3 f_grid(1);    // One block is enough
    
    // Forward pass through convolutional layer C1
    fp_preact_c1<<<c1_grid, c1_block>>>((float (*)[32][3])l_input.output, (float (*)[28][28])l_c1.preact, (float (*)[5][5][3])l_c1.weight);
    fp_bias_c1<<<c1_grid, c1_block>>>((float (*)[28][28])l_c1.preact, l_c1.bias);
    // Using ReLU activation instead of step function
    apply_relu_function<<<c1_grid, c1_block>>>(l_c1.preact, l_c1.output, l_c1.O);

    // Forward pass through subsampling layer S1
    fp_preact_s1<<<s1_grid, s1_block>>>((float (*)[28][28])l_c1.output, (float (*)[7][7])l_s1.preact, (float (*)[4][4])l_s1.weight);
    fp_bias_s1<<<s1_grid, s1_block>>>((float (*)[7][7])l_s1.preact, l_s1.bias);
    // Using ReLU activation instead of step function
    apply_relu_function<<<s1_grid, s1_block>>>(l_s1.preact, l_s1.output, l_s1.O);

    // Forward pass through fully connected layer F
    fp_preact_f<<<f_grid, f_block>>>((float (*)[7][7])l_s1.output, l_f.preact, (float (*)[6][7][7])l_f.weight);
    fp_bias_f<<<f_grid, f_block>>>(l_f.preact, l_f.bias);
    // Using softmax activation for output layer instead of step function
    apply_softmax_function<<<f_grid, f_block>>>(l_f.preact, l_f.output, l_f.O);
    
    // Make sure all kernels complete before measuring time
    cudaDeviceSynchronize();
    
    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Optimized back propagation function
static double back_pass()
{
    clock_t start, end;
    start = clock();

    // Calculate optimal thread and block configuration
    dim3 dimBlock(256); // Use 256 threads per block for better occupancy
    
    // Fully connected layer backpropagation
    // For bp_weight_f: 10*6*7*7 = 2940 total elements
    dim3 dimGrid_f_weight(12); // 12 blocks for 2940 elements with 256 threads/block
    bp_weight_f<<<dimGrid_f_weight, dimBlock>>>((float (*)[6][7][7])l_f.d_weight, l_f.d_preact, (float (*)[7][7])l_s1.output);
    
    // For bp_bias_f: Only 10 elements, can use a single block
    dim3 dimGrid_f_bias(1);
    bp_bias_f<<<dimGrid_f_bias, dimBlock>>>(l_f.bias, l_f.d_preact);

    // Subsampling layer backpropagation
    // For bp_output_s1: 6*7*7 = 294 elements
    dim3 dimGrid_s1_output(2); // 2 blocks for 294 elements
    bp_output_s1<<<dimGrid_s1_output, dimBlock>>>((float (*)[7][7])l_s1.d_output, (float (*)[6][7][7])l_f.weight, l_f.d_preact);
    
    // For bp_preact_s1: 6*7*7 = 294 elements
    dim3 dimGrid_s1_preact(2);
    bp_preact_s1<<<dimGrid_s1_preact, dimBlock>>>((float (*)[7][7])l_s1.d_preact, (float (*)[7][7])l_s1.d_output, (float (*)[7][7])l_s1.preact);
    
    // For bp_weight_s1: 1*4*4 = 16 filter elements
    dim3 dimGrid_s1_weight(16); // 16 blocks, one per filter element
    bp_weight_s1<<<dimGrid_s1_weight, dimBlock>>>((float (*)[4][4])l_s1.d_weight, (float (*)[7][7])l_s1.d_preact, (float (*)[28][28])l_c1.output);
    
    // For bp_bias_s1: Only 1 bias
    dim3 dimGrid_s1_bias(1);
    bp_bias_s1<<<dimGrid_s1_bias, dimBlock>>>(l_s1.bias, (float (*)[7][7])l_s1.d_preact);

    // Convolutional layer backpropagation
    // For bp_output_c1: 6*28*28 = 4704 elements
    dim3 dimGrid_c1_output(19); // 19 blocks for 4704 elements
    bp_output_c1<<<dimGrid_c1_output, dimBlock>>>((float (*)[28][28])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[7][7])l_s1.d_preact);
    
    // For bp_preact_c1: 6*28*28 = 4704 elements
    dim3 dimGrid_c1_preact(19);
    bp_preact_c1<<<dimGrid_c1_preact, dimBlock>>>((float (*)[28][28])l_c1.d_preact, (float (*)[28][28])l_c1.d_output, (float (*)[28][28])l_c1.preact);
    
    // For bp_weight_c1: 6*5*5*3 = 450 filter elements
    dim3 dimGrid_c1_weight(450); // 450 blocks, one block per filter element
    bp_weight_c1<<<dimGrid_c1_weight, dimBlock>>>((float (*)[5][5][3])l_c1.d_weight, (float (*)[28][28])l_c1.d_preact, (float (*)[32][3])l_input.output);
    
    // For bp_bias_c1: 6 biases
    dim3 dimGrid_c1_bias(6); // 6 blocks, one per filter
    bp_bias_c1<<<dimGrid_c1_bias, dimBlock>>>(l_c1.bias, (float (*)[28][28])l_c1.d_preact);

    // Apply gradients - calculate grid size based on total elements
    int f_size = l_f.M * l_f.N;   // Fully connected layer weights
    int s1_size = l_s1.M * l_s1.N; // Subsampling layer weights
    int c1_size = l_c1.M * l_c1.N; // Convolutional layer weights
    
    dim3 dimGrid_f_apply((f_size + dimBlock.x - 1) / dimBlock.x);
    dim3 dimGrid_s1_apply((s1_size + dimBlock.x - 1) / dimBlock.x);
    dim3 dimGrid_c1_apply((c1_size + dimBlock.x - 1) / dimBlock.x);
    
    apply_grad<<<dimGrid_f_apply, dimBlock>>>(l_f.weight, l_f.d_weight, f_size);
    apply_grad<<<dimGrid_s1_apply, dimBlock>>>(l_s1.weight, l_s1.d_weight, s1_size);
    apply_grad<<<dimGrid_c1_apply, dimBlock>>>(l_c1.weight, l_c1.d_weight, c1_size);

    // Make sure all kernels complete before measuring time
    cudaDeviceSynchronize();

    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static void learn()
{
    cublasHandle_t blas;
    cublasCreate(&blas);

    float err;
    const int max_iter = 20;
    int iter = 0;
    
    double time_taken = 0.0;
    
    // Open the CSV file to save training progress
    FILE *csv_file = fopen("training_time_shmem.csv", "w");
    if (!csv_file) {
        fprintf(stderr, "Error: Could not open training_time_shmem.csv for writing\n");
        return;
    }
    
    // Write the CSV header
    fprintf(csv_file, "Epoch,Error,Time_on_GPU\n");

    fprintf(stdout, "Training LeNet on CIFAR-10 using ReLU activation...\n");

    for (iter = 1; iter <= max_iter; ++iter) {
        err = 0.0f;
        double epoch_start_time = time_taken;  // Store starting time for this epoch

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
        fprintf(stdout, "Epoch %d complete, error: %e, time_on_gpu: %lf\n", iter, err, time_taken);
        
        // Write epoch data to CSV
        fprintf(csv_file, "%d,%f,%lf\n", iter, err, time_taken);
        // Flush the file to ensure data is written even if program crashes
        fflush(csv_file);

        if (err < threshold) {
            fprintf(stdout, "Training complete, error less than threshold\n\n");
            break;
        }
    }
    
    // Close the CSV file
    fclose(csv_file);
    
    fprintf(stdout, "\nTotal training time: %lf seconds\n", time_taken);
    fprintf(stdout, "Training data saved to training_time_shmem.csv\n");
    
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