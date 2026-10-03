// Two modes:
//   --diff   loads testdata/patches.bin + testdata/expected.bin (the
//            PyTorch forward pass's own output, from export_weights.py)
//            and reports the max absolute difference between this C++
//            port and PyTorch, over every test patch.
//   --bench  times N_BENCH single-patch inferences and reports p50/p95/p99.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "scorer.hpp"

using Clock = std::chrono::steady_clock;

namespace {

std::vector<float> read_floats(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<float> out(raw.size() / sizeof(float));
    std::memcpy(out.data(), raw.data(), out.size() * sizeof(float));
    return out;
}

int run_diff(const bearing_cnn::Weights& w) {
    auto raw_patches = read_floats("testdata/patches.bin");
    auto expected = read_floats("testdata/expected.bin");
    size_t n = expected.size();
    if (raw_patches.size() != n * bearing_cnn::kPatchLen) {
        std::fprintf(stderr, "patch/expected size mismatch\n");
        return 1;
    }
    double max_abs_diff = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        float out = bearing_cnn::forward(w, raw_patches.data() + i * bearing_cnn::kPatchLen);
        double diff = std::fabs(static_cast<double>(out) - static_cast<double>(expected[i]));
        if (diff > max_abs_diff) { max_abs_diff = diff; worst_idx = i; }
    }
    std::printf("diff: %zu patches, max abs diff = %.9g (at patch %zu)\n", n, max_abs_diff, worst_idx);
    return 0;
}

int run_bench(const bearing_cnn::Weights& w) {
    auto raw_patches = read_floats("testdata/patches.bin");
    size_t n = raw_patches.size() / bearing_cnn::kPatchLen;
    std::vector<double> times_us;
    times_us.reserve(n);
    volatile float sink = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        auto t0 = Clock::now();
        float out = bearing_cnn::forward(w, raw_patches.data() + i * bearing_cnn::kPatchLen);
        auto t1 = Clock::now();
        sink = out;
        times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    (void)sink;
    std::sort(times_us.begin(), times_us.end());
    auto pct = [&](double p) { return times_us[static_cast<size_t>(times_us.size() * p)]; };
    std::printf("bench: %zu inferences, p50=%.4f us, p95=%.4f us, p99=%.4f us, max=%.4f us\n",
                n, pct(0.50), pct(0.95), pct(0.99), times_us.back());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "--diff";
    bearing_cnn::Weights w = bearing_cnn::Weights::load("../weights/perception_weights.bin");
    if (mode == "--diff") return run_diff(w);
    if (mode == "--bench") return run_bench(w);
    std::fprintf(stderr, "usage: %s [--diff|--bench]\n", argv[0]);
    return 1;
}
