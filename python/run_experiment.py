"""Runs both estimators (factor graph, EKF baseline) over N simulated
flights and reports RMS position error through the GNSS outage.

Usage: python run_experiment.py [--n-flights 200] [--out ../docs/estimation_output.txt]
"""
import argparse
import time

import numpy as np
import torch

import config as C
from ekf import run_ekf
from factor_graph import run_factor_graph
from perception_model import load_model
from simulate import generate_flight


def bearing_measurements(flight, model):
    """Runs the trained perception model on every detection's patch, once,
    batched. Returns {(keyframe, landmark_idx): bearing_meas_rad}."""
    if not flight.detections:
        return {}
    patches = np.stack([d.patch for d in flight.detections])
    with torch.no_grad():
        x = torch.from_numpy(patches).float().unsqueeze(1)
        u_pred = model(x).squeeze(1).numpy()
    bearings = u_pred * C.FOV_HALF_ANGLE_RAD
    return {(d.keyframe, d.landmark_idx): float(b) for d, b in zip(flight.detections, bearings)}


def rms_error(p_est, p_true):
    diff = p_est - p_true
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def max_drift(p_est, p_true):
    diff = p_est - p_true
    return float(np.max(np.sqrt(np.sum(diff ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-flights", type=int, default=200)
    ap.add_argument("--out", default="../docs/estimation_output.txt")
    args = ap.parse_args()

    model = load_model("../weights/perception_weights.pt")

    fg_rms = []
    ekf_rms = []
    ekf_max = []
    t0 = time.time()
    for i in range(args.n_flights):
        flight = generate_flight(i)
        bmeas = bearing_measurements(flight, model)
        p_fg = run_factor_graph(flight, bmeas)
        p_ekf = run_ekf(flight, bmeas)
        fg_rms.append(rms_error(p_fg, flight.p_true))
        ekf_rms.append(rms_error(p_ekf, flight.p_true))
        ekf_max.append(max_drift(p_ekf, flight.p_true))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{args.n_flights} flights, {elapsed:.1f}s elapsed, "
                  f"running FG mean RMS so far {np.mean(fg_rms):.3f} m")

    elapsed = time.time() - t0
    lines = []
    lines.append(f"N_FLIGHTS={args.n_flights}, elapsed={elapsed:.1f}s")
    lines.append(f"Factor graph: mean RMS position error over {args.n_flights} flights = "
                 f"{np.mean(fg_rms):.4f} m (std {np.std(fg_rms):.4f}, max {np.max(fg_rms):.4f})")
    lines.append(f"EKF baseline: mean RMS position error over {args.n_flights} flights = "
                 f"{np.mean(ekf_rms):.4f} m (std {np.std(ekf_rms):.4f})")
    lines.append(f"EKF baseline: mean of per-flight max drift = {np.mean(ekf_max):.4f} m "
                 f"(max over all flights {np.max(ekf_max):.4f} m)")
    out = "\n".join(lines)
    print(out)
    with open(args.out, "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
