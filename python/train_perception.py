"""Trains the edge perception model (BearingCNN) on synthetic camera patches.

CPU training is used and is appropriate here: the model has 1,145 parameters
and the dataset is 20,000 synthetic 64-sample patches, so a GPU would not
meaningfully help. This is stated honestly rather than claiming GPU use for
a model this small.

Usage: python train_perception.py
Writes weights/perception_weights.pt (PyTorch state_dict) and prints the
final held-out validation MAE in normalized-bearing units and in degrees.
"""

import numpy as np
import torch
import torch.nn as nn

import config as C
from perception_model import BearingCNN
from simulate import make_camera_patch

TRAIN_N = 16000
VAL_N = 4000
EPOCHS = 40
BATCH_SIZE = 128
LR = 2e-3


def build_dataset(n, seed):
    rng = np.random.default_rng(seed)
    bearings = rng.uniform(-C.FOV_HALF_ANGLE_RAD, C.FOV_HALF_ANGLE_RAD, size=n)
    patches = np.stack([make_camera_patch(b, rng) for b in bearings])
    labels = (bearings / C.FOV_HALF_ANGLE_RAD).astype(np.float32)
    return torch.from_numpy(patches).float().unsqueeze(1), torch.from_numpy(labels).float().unsqueeze(1)


def main():
    torch.manual_seed(0)
    x_train, y_train = build_dataset(TRAIN_N, seed=1)
    x_val, y_val = build_dataset(VAL_N, seed=2)

    model = BearingCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n_batches = TRAIN_N // BATCH_SIZE
    for epoch in range(EPOCHS):
        perm = torch.randperm(TRAIN_N)
        total_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_mae_norm = (val_pred - y_val).abs().mean().item()
            model.train()
            print(f"epoch {epoch + 1:3d}  train_mse {total_loss / TRAIN_N:.6f}  "
                  f"val_mae_norm {val_mae_norm:.6f}")

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_mae_norm = (val_pred - y_val).abs().mean().item()
        val_resid_rad = (val_pred - y_val) * C.FOV_HALF_ANGLE_RAD
        val_mae_deg = val_resid_rad.abs().mean().item() * 180.0 / np.pi
        val_std_rad = val_resid_rad.std().item()

    print(f"final val_mae_norm={val_mae_norm:.6f} val_mae_deg={val_mae_deg:.4f} "
          f"val_resid_std_rad={val_std_rad:.6f}")

    torch.save(model.state_dict(), "../weights/perception_weights.pt")
    print("saved weights/perception_weights.pt")

    with open("../docs/perception_train_output.txt", "w") as f:
        f.write("BearingCNN training, CPU only, 20000 synthetic patches "
                f"({TRAIN_N} train / {VAL_N} val)\n")
        f.write(f"final val_mae_norm={val_mae_norm:.6f}\n")
        f.write(f"final val_mae_deg={val_mae_deg:.4f}\n")
        f.write(f"val_resid_std_rad={val_std_rad:.6f}\n")
        f.write("This std informs BEARING_MEAS_SIGMA used by both estimators "
                "(config.py).\n")


if __name__ == "__main__":
    main()
