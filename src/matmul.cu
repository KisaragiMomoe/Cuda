#include "kernels.h"

#define TILE_SIZE 32
#define THREAD_TILE 4

// ============ 核函数：Tiled MatMul（寄存器分块 + float4） ============
__global__ void __launch_bounds__(64)
matmul_kernel(const float* __restrict__ A,
              const float* __restrict__ B,
              float* __restrict__ C,
              int M, int K, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float sB[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int rowBase = ty * THREAD_TILE;
    int colBase = tx * THREAD_TILE;

    int globalRow = blockIdx.y * TILE_SIZE + rowBase;
    int globalCol = blockIdx.x * TILE_SIZE + colBase;

    float c[THREAD_TILE][THREAD_TILE] = {0.0f};
    float aReg[THREAD_TILE];
    float bReg[THREAD_TILE];

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // --- 向量化加载 A 和 B 的块 ---
        for (int i = ty; i < TILE_SIZE; i += blockDim.y) {
            for (int j = tx * 4; j < TILE_SIZE; j += blockDim.x * 4) {
                // 加载 A
                int aRow = blockIdx.y * TILE_SIZE + i;
                int aCol = t * TILE_SIZE + j;
                if (aRow < M && aCol + 3 < K) {
                    float4 tmp = *reinterpret_cast<const float4*>(&A[aRow * K + aCol]);
                    sA[i][j]     = tmp.x;
                    sA[i][j + 1] = tmp.y;
                    sA[i][j + 2] = tmp.z;
                    sA[i][j + 3] = tmp.w;
                } else {
                    for (int d = 0; d < 4; d++) {
                        int col = aCol + d;
                        sA[i][j + d] = (aRow < M && col < K) ? A[aRow * K + col] : 0.0f;
                    }
                }

                // 加载 B
                int bRow = t * TILE_SIZE + i;
                int bCol = blockIdx.x * TILE_SIZE + j;
                if (bRow < K && bCol + 3 < N) {
                    float4 tmp = *reinterpret_cast<const float4*>(&B[bRow * N + bCol]);
                    sB[i][j]     = tmp.x;
                    sB[i][j + 1] = tmp.y;
                    sB[i][j + 2] = tmp.z;
                    sB[i][j + 3] = tmp.w;
                } else {
                    for (int d = 0; d < 4; d++) {
                        int col = bCol + d;
                        sB[i][j + d] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        // --- 寄存器复用计算 ---
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < THREAD_TILE; i++) aReg[i] = sA[rowBase + i][k];
            for (int j = 0; j < THREAD_TILE; j++) bReg[j] = sB[k][colBase + j];
            for (int i = 0; i < THREAD_TILE; i++)
                for (int j = 0; j < THREAD_TILE; j++)
                    c[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

    // --- 写回结果 ---
    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            int r = globalRow + i;
            int cc = globalCol + j;
            if (r < M && cc < N) C[r * N + cc] = c[i][j];
        }
    }
}

// ============ 封装函数：供外部调用 ============
void launchMatMul(const float* A, const float* B, float* C,
                  int M, int K, int N) {
    dim3 block(8, 8);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
}