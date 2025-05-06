#include "cifar_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global variable definitions
cifar_data *train_set, *test_set;
unsigned int train_cnt, test_cnt;

// Internal function to load data from binary files
void load_cifar10(const char *data_dir, cifar_data **data_set, unsigned int *data_count, const char *batch_files[], int num_batches) {
    const int images_per_batch = 10000;
    *data_count = images_per_batch * num_batches;
    *data_set = (cifar_data *)malloc(sizeof(cifar_data) * (*data_count));

    int img_idx = 0;
    for (int batch = 0; batch < num_batches; batch++) {
        char filename[256];
        sprintf(filename, "%s/%s", data_dir, batch_files[batch]);

        FILE *fp = fopen(filename, "rb");
        if (!fp) {
            fprintf(stderr, "Failed to open %s\n", filename);
            exit(1);
        }

        for (int i = 0; i < images_per_batch; i++) {
            fread(&((*data_set)[img_idx].label), sizeof(unsigned char), 1, fp);

            unsigned char buffer[CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS];
            fread(buffer, sizeof(unsigned char), CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS, fp);

            for (int c = 0; c < CIFAR_CHANNELS; c++) {
                for (int h = 0; h < CIFAR_HEIGHT; h++) {
                    for (int w = 0; w < CIFAR_WIDTH; w++) {
                        int offset = c * CIFAR_WIDTH * CIFAR_HEIGHT + h * CIFAR_WIDTH + w;
                        (*data_set)[img_idx].data[h][w][c] = buffer[offset] / 255.0f;
                    }
                }
            }

            img_idx++;
        }

        fclose(fp);
    }
}

// Load both training and test sets
void load_cifar_data(const char *data_dir) {
    const char *train_batch_files[5] = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
    };

    load_cifar10(data_dir, &train_set, &train_cnt, train_batch_files, 5);

    const char *test_batch_files[1] = {
        "test_batch.bin"
    };

    load_cifar10(data_dir, &test_set, &test_cnt, test_batch_files, 1);

    printf("Loaded %u training images and %u test images\n", train_cnt, test_cnt);
}
