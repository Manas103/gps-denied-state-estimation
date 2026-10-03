// Hand-ported inference of python/perception_model.py's BearingCNN:
// conv1d(1->8, k=5, pad=2) -> ReLU -> conv1d(8->16, k=5, pad=2) -> ReLU ->
// global average pool over length -> fc(16->16) -> ReLU -> fc(16->1).
//
// This is a direct translation, not a call into a tensor library: each
// conv1d is a hand-written sliding-window loop over Eigen vectors, chosen
// deliberately (see README) so the C++ side is a checkable line-for-line
// port of the PyTorch module, not a black box that merely produces similar
// numbers.
#pragma once

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

namespace bearing_cnn {

constexpr int kPatchLen = 64;
constexpr int kConv1Out = 8;
constexpr int kConv2Out = 16;
constexpr int kKernel = 5;
constexpr int kPad = 2;
constexpr int kFcHidden = 16;

struct Weights {
    // conv1: (8, 1, 5) -> stored as 8 vectors of length 5
    Eigen::Matrix<float, kConv1Out, kKernel> conv1_w;
    Eigen::Matrix<float, kConv1Out, 1> conv1_b;
    // conv2: (16, 8, 5) -> 16 x 8 x 5
    std::array<Eigen::Matrix<float, kConv2Out, kKernel>, kConv1Out> conv2_w;  // indexed [in_ch]
    Eigen::Matrix<float, kConv2Out, 1> conv2_b;
    Eigen::Matrix<float, kFcHidden, kConv2Out> fc1_w;
    Eigen::Matrix<float, kFcHidden, 1> fc1_b;
    Eigen::Matrix<float, 1, kFcHidden> fc2_w;
    float fc2_b;

    // Reads the flat float32 layout export_weights.py writes: conv1.weight
    // (8*1*5), conv1.bias(8), conv2.weight(16*8*5), conv2.bias(16),
    // fc1.weight(16*16), fc1.bias(16), fc2.weight(1*16), fc2.bias(1).
    static Weights load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open weights file: " + path);
        // istreambuf_iterator<char> into vector<float> would value-convert
        // each byte (0-255) into a float instead of reinterpreting 4 bytes
        // as an IEEE-754 float; read raw bytes first, then reinterpret.
        std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<float> buf(raw.size() / sizeof(float));
        std::memcpy(buf.data(), raw.data(), buf.size() * sizeof(float));
        size_t expect = kConv1Out * 1 * kKernel + kConv1Out
                      + kConv2Out * kConv1Out * kKernel + kConv2Out
                      + kFcHidden * kConv2Out + kFcHidden
                      + 1 * kFcHidden + 1;
        if (buf.size() != expect) {
            throw std::runtime_error("weights file has unexpected size");
        }
        const float* p = buf.data();
        Weights w;
        for (int o = 0; o < kConv1Out; ++o)
            for (int k = 0; k < kKernel; ++k) w.conv1_w(o, k) = *p++;
        for (int o = 0; o < kConv1Out; ++o) w.conv1_b(o) = *p++;
        for (int o = 0; o < kConv2Out; ++o)
            for (int ic = 0; ic < kConv1Out; ++ic)
                for (int k = 0; k < kKernel; ++k) w.conv2_w[ic](o, k) = *p++;
        for (int o = 0; o < kConv2Out; ++o) w.conv2_b(o) = *p++;
        for (int o = 0; o < kFcHidden; ++o)
            for (int ic = 0; ic < kConv2Out; ++ic) w.fc1_w(o, ic) = *p++;
        for (int o = 0; o < kFcHidden; ++o) w.fc1_b(o) = *p++;
        for (int ic = 0; ic < kFcHidden; ++ic) w.fc2_w(0, ic) = *p++;
        w.fc2_b = *p++;
        return w;
    }
};

// One patch (kPatchLen samples) in, one raw scalar output ("logit" in the
// loose sense the Python side also uses it: the model's final unactivated
// scalar), matching python/perception_model.py::BearingCNN.forward exactly,
// including the padding convention (PyTorch's Conv1d with padding=2, which
// zero-pads both ends before sliding the kernel).
inline float forward(const Weights& w, const float* patch) {
    // conv1: 1 input channel -> 8 output channels, same-length output via
    // zero padding.
    float c1[kConv1Out][kPatchLen];
    for (int o = 0; o < kConv1Out; ++o) {
        for (int t = 0; t < kPatchLen; ++t) {
            float acc = w.conv1_b(o);
            for (int k = 0; k < kKernel; ++k) {
                int in_t = t + k - kPad;
                if (in_t >= 0 && in_t < kPatchLen) acc += w.conv1_w(o, k) * patch[in_t];
            }
            c1[o][t] = std::max(0.0f, acc);  // ReLU
        }
    }

    // conv2: 8 input channels -> 16 output channels.
    float c2[kConv2Out][kPatchLen];
    for (int o = 0; o < kConv2Out; ++o) {
        for (int t = 0; t < kPatchLen; ++t) {
            float acc = w.conv2_b(o);
            for (int ic = 0; ic < kConv1Out; ++ic) {
                for (int k = 0; k < kKernel; ++k) {
                    int in_t = t + k - kPad;
                    if (in_t >= 0 && in_t < kPatchLen) acc += w.conv2_w[ic](o, k) * c1[ic][in_t];
                }
            }
            c2[o][t] = std::max(0.0f, acc);
        }
    }

    // Global average pool over length -> (16,).
    Eigen::Matrix<float, kConv2Out, 1> pooled;
    for (int o = 0; o < kConv2Out; ++o) {
        float sum = 0.0f;
        for (int t = 0; t < kPatchLen; ++t) sum += c2[o][t];
        pooled(o) = sum / static_cast<float>(kPatchLen);
    }

    Eigen::Matrix<float, kFcHidden, 1> h = (w.fc1_w * pooled + w.fc1_b).cwiseMax(0.0f);
    float out = (w.fc2_w * h)(0, 0) + w.fc2_b;
    return out;
}

}  // namespace bearing_cnn
