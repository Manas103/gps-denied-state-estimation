"""Exports BearingCNN's trained weights to a flat float32 binary the C++
scorer (cpp/scorer.cpp) reads directly, and generates a fixed set of test
patches plus their PyTorch forward-pass outputs for the C++/PyTorch diff
harness.

Weight file layout (all float32, little-endian, in this exact order):
  conv1.weight (8*1*5), conv1.bias (8)
  conv2.weight (16*8*5), conv2.bias (16)
  fc1.weight (16*16), fc1.bias (16)
  fc2.weight (1*16), fc2.bias (1)

Usage: python export_weights.py
Writes weights/perception_weights.bin, cpp/testdata/patches.bin,
cpp/testdata/expected.bin.
"""
import os

import numpy as np
import torch

import config as C
from perception_model import load_model
from simulate import make_camera_patch

N_TEST_PATCHES = 2000


def main():
    model = load_model("../weights/perception_weights.pt")
    sd = model.state_dict()
    order = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
             "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    flat = np.concatenate([sd[k].numpy().astype(np.float32).ravel() for k in order])
    flat.tofile("../weights/perception_weights.bin")
    print(f"wrote weights/perception_weights.bin ({flat.nbytes} bytes, {flat.size} floats)")

    rng = np.random.default_rng(20260911)
    bearings = rng.uniform(-C.FOV_HALF_ANGLE_RAD, C.FOV_HALF_ANGLE_RAD, size=N_TEST_PATCHES)
    patches = np.stack([make_camera_patch(b, rng) for b in bearings]).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(patches).unsqueeze(1)
        expected = model(x).squeeze(1).numpy().astype(np.float32)

    os.makedirs("../cpp/testdata", exist_ok=True)
    patches.tofile("../cpp/testdata/patches.bin")
    expected.tofile("../cpp/testdata/expected.bin")
    print(f"wrote cpp/testdata/patches.bin and expected.bin ({N_TEST_PATCHES} patches, "
          f"{C.CAMERA_PATCH_LEN} samples each)")


if __name__ == "__main__":
    main()
