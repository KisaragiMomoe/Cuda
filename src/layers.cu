#include"layers.h"
#include"kernels.h"
#include<cstdio>
#include<cstdlib>

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    int batch = input.rows;
    int in_features = input.cols;
    int out_features = weight.rows;
    
    // --- 1. 权重转置：weight [out x in] → weight_T [in x out] ---
    // 为了教学简单，先在 CPU 端做转置
    int w_size = in_features * out_features;
    float* h_w_origin = (float*)malloc(w_size * sizeof(float));
    float* h_w_trans = (float*)malloc(w_size * sizeof(float));
    weight.copyToHost(h_w_origin);
    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < in_features; j++) {
            h_w_trans[j * out_features + i] = h_w_origin[i * in_features + j];
        }
    }
    Tensor weight_T(in_features, out_features);
    weight_T.copyFromHost(h_w_trans);
    free(h_w_origin), free(h_w_trans);
    Tensor output(batch, out_features);
    launchMatMul(input.data, weight_T.data, output.data, batch, in_features, out_features);
    launchAddBias(output.data, bias.data, batch, out_features);
    return output;
}

Tensor mlp_forward (const Tensor& input, const Tensor& w1, const Tensor& b1, const Tensor& w2, const Tensor& b2) {
    printf("  [MLP] Linear 1: [%d x %d] × [%d x %d]^T + bias\n",
           input.rows, input.cols, w1.rows, w1.cols);
    Tensor hidden = linear(input, w1, b1);

    printf("  [MLP] ReLU: [%d x %d]\n", hidden.rows, hidden.cols);
    Tensor hidden_relu(hidden.rows, hidden.cols);
    launchReLU(hidden.data, hidden_relu.data, hidden.rows * hidden.cols);

    // 第 2 层：Linear
    printf("  [MLP] Linear 2: [%d x %d] × [%d x %d]^T + bias\n",
           hidden_relu.rows, hidden_relu.cols, w2.rows, w2.cols);
    Tensor logits = linear(hidden_relu, w2, b2);

    // Softmax
    printf("  [MLP] Softmax: [%d x %d]\n", logits.rows, logits.cols);
    Tensor probs(logits.rows, logits.cols);
    launchSoftmax(logits.data, probs.data, logits.rows, logits.cols);

    return probs;
}