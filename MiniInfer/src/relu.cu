#include "kernels.h"

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = fmaxf(input[i], 0.0f);
    return ;
}

void launchReLU(const float* input, float* output, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_kernel<<<grid, block>>>(input, output, n);
    return ;
}

__global__ void add_bias_kernal(float* __restrict__ data, const float* __restrict__ bias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col < cols && row < rows) data[cols * row + col] += bias[col];
    return ;
}

void launchAddBias(float* data, const float* bias, int rows, int cols) {
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    add_bias_kernal<<<grid, block>>>(data, bias, rows, cols);
    return ;
}