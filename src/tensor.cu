#include "tensor.h"

// ============ 构造函数：分配显存 ============
Tensor :: Tensor(int r, int c) : rows(r), cols(c) {
    size_t siz = (size_t)rows * cols * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&data, siz);
    if (err != cudaSuccess) {
        printf("Tensor cudaMalloc Failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    return ;
}

// ============ 析构函数：释放显存 ============
Tensor :: ~Tensor() {
    if (data) cudaFree(data);
    return ;
}

// ============ 移动构造函数 ============
Tensor :: Tensor(Tensor&& other) noexcept : data(other.data), rows(other.rows), cols(other.cols) {
    other.data = nullptr;
    return ;
}

// ============ 主机 → 设备 ============
void Tensor :: copyFromHost(const float* h_data) {
    size_t siz = (size_t)rows * cols * sizeof(float);
    cudaMemcpy(data, h_data, siz, cudaMemcpyHostToDevice);
    return ;
}

// ============ 设备 → 主机 ============
void Tensor :: copyToHost(float* h_data) const {
    size_t siz = (size_t)rows * cols * sizeof(float);
    cudaMemcpy(h_data, data, siz, cudaMemcpyDeviceToHost);
    return ;
}

// ============ 调试打印 ============
void Tensor :: print(const char* name) const {
    int n = rows * cols;
    float* h = (float*)malloc(n * sizeof(float));
    copyToHost(h);
    printf("%s [%d x %d]: ", name, rows, cols);
    int limit = (n < 8) ? n : 8;
    for (int i = 0; i < limit; i++) printf("%.4f ", h[i]);
    if (n > 8) printf("...");
    printf("\n");
    free(h);
    return ;
}