#pragma once

#include<cstdio>
#include<cstdlib>
#include<cuda_runtime.h>

struct Tensor {
    float* data; // 指向设备端显存的指针
    int rows, cols; // 行数 列数
    Tensor(int r, int c); // 构造函数：根据 rows × cols 分配显存
    ~Tensor(); // 析构函数：自动释放显存
    Tensor(const Tensor&) = delete; // 禁止拷贝（防止两个 Tensor 指向同一块显存，导致双重释放）
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept; // 允许移动语义（把所有权从一个 Tensor 转移给另一个）
    void copyFromHost(const float* h_data); // 主机 → 设备：把 CPU 数据拷到 GPU
    void copyToHost(float* h_data) const; // 设备 → 主机：把 GPU 数据拷回 CPU
    void print(const char* name) const; // 打印前几个元素（调试用）
};