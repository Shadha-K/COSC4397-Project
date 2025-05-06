#ifndef CIFAR_LOADER_H
#define CIFAR_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CIFAR_WIDTH 32
#define CIFAR_HEIGHT 32
#define CIFAR_CHANNELS 3
#define CIFAR_CLASSES 10

typedef struct {
    float data[CIFAR_HEIGHT][CIFAR_WIDTH][CIFAR_CHANNELS];
    unsigned char label;
} cifar_data;

// Declare global variables (actual definition is in .c file)
extern cifar_data *train_set, *test_set;
extern unsigned int train_cnt, test_cnt;

// Function declarations only
void load_cifar10(const char *data_dir, cifar_data **data_set, unsigned int *data_count, const char *batch_files[], int num_batches);
void load_cifar_data(const char *data_dir);

#endif // CIFAR_LOADER_H
