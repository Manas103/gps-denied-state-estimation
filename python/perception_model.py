"""The edge perception model: a tiny 1D CNN bearing detector.

Input: a CAMERA_PATCH_LEN-sample synthetic camera scanline (see
simulate.make_camera_patch). Output: a single scalar, the estimated
normalized bearing u = bearing / FOV_HALF_ANGLE_RAD, in roughly [-1, 1].

This is the model that gets hand-ported to Eigen-based C++ in
cpp/scorer.cpp and diffed exactly against this PyTorch forward pass
(see export_weights.py and docs/oracle_diff_output.txt). The architecture
is deliberately small (two 1D conv layers, global average pool, two FC
layers) so the C++ port is a faithful, checkable hand-translation rather
than a call into a generic tensor library.
"""

import torch
import torch.nn as nn

import config as C


class BearingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x: (B, 1, L)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=2)          # global average pool over length -> (B, 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)            # (B, 1) raw output ("logit" in the loose
                                    # sense used throughout this repo: the
                                    # model's final unactivated scalar output)
        return x


def load_model(weights_path):
    model = BearingCNN()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
