#pragma once
#include"tensor.h"

// Linear 层：output = input × weight^T + bias
// input:  [batch x in_features]
// weight: [out_features x in_features]
// bias:   [1 x out_features]
// 返回:   [batch x out_features]
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);
// 两层 MLP + Softmax
// input: [batch x in_features]
// W1:    [hidden x in_features]
// b1:    [1 x hidden]
// W2:    [out_features x hidden]
// b2:    [1 x out_features]
// 返回:  [batch x out_features]（每行和 = 1.0）
Tensor mlp_forward(const Tensor& input, const Tensor& W1, const Tensor& b1, const Tensor& W2, const Tensor& b2);
