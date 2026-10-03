// Hand-rolled check suite for the hand-ported BearingCNN forward pass.
// The real correctness claim (max abs diff vs PyTorch) is measured by
// scorer_main --diff against docs/oracle_diff_output.txt; these are the
// structural properties that would catch a broken port even without the
// PyTorch reference file present.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../scorer.hpp"

static int g_checks = 0;
#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); std::exit(1); } \
} while (0)

int main() {
    bearing_cnn::Weights w = bearing_cnn::Weights::load("../weights/perception_weights.bin");

    // Determinism: the same patch must produce the same output every time
    // (no hidden state, no uninitialized reads).
    float patch[bearing_cnn::kPatchLen];
    for (int i = 0; i < bearing_cnn::kPatchLen; ++i) patch[i] = 0.1f * static_cast<float>(i % 7);
    float a = bearing_cnn::forward(w, patch);
    float b = bearing_cnn::forward(w, patch);
    CHECK(a == b);

    // A finite, non-NaN output for an all-zero patch (a degenerate but
    // legal input: no bump at all).
    float zero_patch[bearing_cnn::kPatchLen] = {0};
    float z = bearing_cnn::forward(w, zero_patch);
    CHECK(std::isfinite(z));

    // Sensitivity: shifting the bump's location must change the output.
    // Two patches with a bump at opposite ends should not score identically.
    float left[bearing_cnn::kPatchLen], right[bearing_cnn::kPatchLen];
    for (int i = 0; i < bearing_cnn::kPatchLen; ++i) {
        left[i] = std::exp(-0.5f * static_cast<float>((i - 8) * (i - 8)) / 4.0f);
        right[i] = std::exp(-0.5f * static_cast<float>((i - 56) * (i - 56)) / 4.0f);
    }
    float out_left = bearing_cnn::forward(w, left);
    float out_right = bearing_cnn::forward(w, right);
    CHECK(std::fabs(out_left - out_right) > 1e-4f);

    // Loading a nonexistent weights file must throw, not crash or silently
    // zero-initialize.
    bool threw = false;
    try {
        bearing_cnn::Weights::load("does_not_exist.bin");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    std::printf("check_scorer: %d checks passed\n", g_checks);
    return 0;
}
