#pragma once

#include<cuda_runtime.h>

void launchMatMul(const float* A, const float* B, float* C, int M, int K, int N);

void launchSoftmax(const float* input, float* output, int rows, int cols);

void launchReLU(const float* input, float* output, int n);

void launchAddBias(float* data, const float* bias, int rows, int cols);