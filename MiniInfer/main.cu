#include "tensor.h"
#include "kernels.h"
#include "layers.h"
#include <cstdio>
#include <cmath>

int main() {
    printf("=== MiniInfer 第 5 步：完整 MLP 推理 ===\n\n");

    const int BATCH = 128;
    const int IN_FEATURES = 256;
    const int HIDDEN = 512;
    const int OUT_FEATURES = 1024;  // 为了复用 Softmax

    printf("网络结构：\n");
    printf("  输入:   [%d x %d]\n", BATCH, IN_FEATURES);
    printf("  隐层:   [%d x %d]\n", BATCH, HIDDEN);
    printf("  输出:   [%d x %d]\n\n", BATCH, OUT_FEATURES);

    // --- 1. 主机端准备数据 ---
    float* h_input = (float*)malloc(BATCH * IN_FEATURES * sizeof(float));
    float* h_W1 = (float*)malloc(HIDDEN * IN_FEATURES * sizeof(float));
    float* h_b1 = (float*)malloc(HIDDEN * sizeof(float));
    float* h_W2 = (float*)malloc(OUT_FEATURES * HIDDEN * sizeof(float));
    float* h_b2 = (float*)malloc(OUT_FEATURES * sizeof(float));

    for (int i = 0; i < BATCH * IN_FEATURES; i++) h_input[i] = 0.01f * (i % 100);
    for (int i = 0; i < HIDDEN * IN_FEATURES; i++) h_W1[i] = 0.001f * (i % 50);
    for (int i = 0; i < HIDDEN; i++) h_b1[i] = 0.0f;
    for (int i = 0; i < OUT_FEATURES * HIDDEN; i++) h_W2[i] = 0.001f * (i % 30);
    for (int i = 0; i < OUT_FEATURES; i++) h_b2[i] = 0.0f;

    // --- 2. 上传到 GPU ---
    Tensor d_input(BATCH, IN_FEATURES);
    Tensor d_W1(HIDDEN, IN_FEATURES);
    Tensor d_b1(1, HIDDEN);
    Tensor d_W2(OUT_FEATURES, HIDDEN);
    Tensor d_b2(1, OUT_FEATURES);

    d_input.copyFromHost(h_input);
    d_W1.copyFromHost(h_W1);
    d_b1.copyFromHost(h_b1);
    d_W2.copyFromHost(h_W2);
    d_b2.copyFromHost(h_b2);

    // --- 3. 前向推理 ---
    printf("开始前向推理...\n");
    Tensor probs = mlp_forward(d_input, d_W1, d_b1, d_W2, d_b2);
    printf("前向推理完成\n\n");

    // --- 4. 验证：每行的概率之和应该为 1 ---
    float* h_probs = (float*)malloc(BATCH * OUT_FEATURES * sizeof(float));
    probs.copyToHost(h_probs);

    int errors = 0;
    for (int r = 0; r < BATCH; r++) {
        float sum = 0.0f;
        for (int c = 0; c < OUT_FEATURES; c++) {
            sum += h_probs[r * OUT_FEATURES + c];
        }
        if (fabsf(sum - 1.0f) > 1e-3) {
            if (errors < 3) {
                printf("第 %d 行：sum = %.6f\n", r, sum);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("✅ 全部 %d 行验证通过（每行概率和为 1.0）\n", BATCH);
    } else {
        printf("❌ 共有 %d 行失败\n", errors);
    }

    free(h_input); free(h_W1); free(h_b1); free(h_W2); free(h_b2); free(h_probs);
    return 0;
}