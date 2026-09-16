#include "kernels.h"
#include <math.h>

#define BLOCK_SIZE 256

// ============ 核函数：每行一个 Block，Warp Shuffle 归约 + float4 ============
__global__ void softmax_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // --- 1. float4 加载 ---
    const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
    float4 v = row_input4[tid];
    float reg[4] = {v.x, v.y, v.z, v.w};

    // --- 2. 局部最大值 ---
    float local_max = reg[0];
    for (int i = 1; i < 4; i++) local_max = fmaxf(local_max, reg[i]);

    // --- 3. Warp Shuffle 归约求全局最大值 ---
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    __shared__ float sdata[BLOCK_SIZE / 32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) sdata[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v2 = (lane_id < BLOCK_SIZE / 32) ? sdata[lane_id] : -INFINITY;
        for (int offset = 16; offset > 0; offset >>= 1)
            v2 = fmaxf(v2, __shfl_down_sync(0xffffffff, v2, offset));
        if (lane_id == 0) sdata[0] = v2;
    }
    __syncthreads();
    float row_max = sdata[0];
    __syncthreads();

    // --- 4. 计算 exp 并求和 ---
    float local_sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        reg[i] = __expf(reg[i] - row_max);
        local_sum += reg[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    if (lane_id == 0) sdata[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v2 = (lane_id < BLOCK_SIZE / 32) ? sdata[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v2 += __shfl_down_sync(0xffffffff, v2, offset);
        if (lane_id == 0) sdata[0] = v2;
    }
    __syncthreads();
    float inv_sum = 1.0f / sdata[0];

    // --- 5. float4 写回 ---
    float4 out;
    out.x = reg[0] * inv_sum;
    out.y = reg[1] * inv_sum;
    out.z = reg[2] * inv_sum;
    out.w = reg[3] * inv_sum;
    reinterpret_cast<float4*>(row_output)[tid] = out;
}

// ============ 封装函数 ============
void launchSoftmax(const float* input, float* output, int rows, int cols) {
    // 暂时只支持 cols == 1024
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);
    softmax_kernel<<<grid, block>>>(input, output, cols);
}