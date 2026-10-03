"""Sliding-window factor graph fusing simulated IMU, barometer and camera
(bearing, via the trained edge perception model) measurements.

Each node holds a 9-vector state [p(3), v(3), ba(3)]: position, velocity and
accelerometer bias. Consecutive nodes are linked by an IMU (preintegration)
factor; every node with a barometer sample gets an altitude factor; every
node with a camera detection gets a bearing factor, where the "measurement"
is the trained BearingCNN's output on the detection's patch, never the
simulator's true bearing.

The window is solved with scipy.optimize.least_squares (Levenberg-Marquardt),
which is the Gauss-Newton family algorithm FG_GN_ITERATIONS in config.py
names; this file does not hand-roll its own normal-equations solver, and
says so rather than implying otherwise (see README "what this is NOT").

Marginalization of nodes that leave the window is approximated by a fixed
Gaussian prior on the window's oldest node, carried forward from that node's
own solved estimate the last time it was the newest node in a window. This
is cheaper than a real Schur-complement marginalization and is disclosed as
a simplification in the README.
"""
import numpy as np
from scipy.optimize import least_squares

import config as C


def _wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


PRIOR_POS_SIGMA = 0.6
PRIOR_VEL_SIGMA = 0.15
PRIOR_BIAS_SIGMA = 0.02


def _imu_residual(x_i, x_ip1, accel_meas, dt):
    p_i, v_i, ba_i = x_i[0:3], x_i[3:6], x_i[6:9]
    p_j, v_j, ba_j = x_ip1[0:3], x_ip1[3:6], x_ip1[6:9]
    a = accel_meas - ba_i
    p_pred = p_i + v_i * dt + 0.5 * a * dt * dt
    v_pred = v_i + a * dt
    r_p = (p_j - p_pred) / C.ACCEL_WHITE_NOISE_SIGMA_P_EQUIV
    r_v = (v_j - v_pred) / C.ACCEL_WHITE_NOISE_SIGMA_V_EQUIV
    r_ba = (ba_j - ba_i) / (C.ACCEL_BIAS_RW_SIGMA * np.sqrt(dt))
    return np.concatenate([r_p, r_v, r_ba])


def _residuals(x_flat, window_nodes, prior_x, accel_meas_avg, baro_meas, detections_by_node,
               model, dt):
    n = len(window_nodes)
    X = x_flat.reshape(n, 9)
    res = []

    # Prior on the oldest node in the window.
    d0 = X[0] - prior_x
    res.append(d0[0:3] / PRIOR_POS_SIGMA)
    res.append(d0[3:6] / PRIOR_VEL_SIGMA)
    res.append(d0[6:9] / PRIOR_BIAS_SIGMA)

    # IMU factors between consecutive nodes in the window.
    for i in range(n - 1):
        node_i = window_nodes[i]
        res.append(_imu_residual(X[i], X[i + 1], accel_meas_avg[node_i], dt))

    # Barometer factors.
    for i, node in enumerate(window_nodes):
        res.append(np.array([(X[i, 2] - baro_meas[node]) / C.BARO_NOISE_SIGMA]))

    # Bearing factors (perception-model output, never the true bearing).
    for i, node in enumerate(window_nodes):
        for det in detections_by_node.get(node, []):
            px, py = X[i, 0], X[i, 1]
            L = C.LANDMARKS[det.landmark_idx]
            dx, dy = L[0] - px, L[1] - py
            bearing_pred = _wrap(np.arctan2(dy, dx) - det.yaw)
            r = _wrap(bearing_pred - det.bearing_meas) / C.BEARING_MEAS_SIGMA
            res.append(np.array([r]))

    return np.concatenate(res)


def run_factor_graph(flight, bearing_meas_by_det):
    """bearing_meas_by_det: dict keyed by (keyframe, landmark_idx) -> bearing_meas (rad),
    already produced by the perception model, not the simulator's ground truth."""
    n_nodes = flight.n_keyframes + 1
    dt = flight.dt_keyframe

    detections_by_node = {}
    for det in flight.detections:
        det.bearing_meas = bearing_meas_by_det[(det.keyframe, det.landmark_idx)]
        det.yaw = flight.yaw_nodes[det.keyframe]
        detections_by_node.setdefault(det.keyframe, []).append(det)

    p0 = flight.p_true[0] + flight.p0_est_error
    v0 = flight.v_true[0] + flight.v0_est_error
    ba0 = flight.ba_true[0] + flight.ba0_est_error
    x_est = {0: np.concatenate([p0, v0, ba0])}

    p_history = np.zeros((n_nodes, 3))
    p_history[0] = p0

    for k in range(1, n_nodes):
        lo = max(0, k - C.FG_WINDOW_SIZE + 1)
        window_nodes = list(range(lo, k + 1))

        # Seed every node in the window: known nodes from x_est, new ones by
        # propagating the IMU factor forward from the last known node.
        seed = []
        for node in window_nodes:
            if node in x_est:
                seed.append(x_est[node])
            else:
                prev = seed[-1]
                a = flight.accel_meas_avg[node - 1] - prev[6:9]
                p_pred = prev[0:3] + prev[3:6] * dt + 0.5 * a * dt * dt
                v_pred = prev[3:6] + a * dt
                seed.append(np.concatenate([p_pred, v_pred, prev[6:9]]))
        x0 = np.concatenate(seed)
        prior_x = x_est[lo]

        result = least_squares(
            _residuals, x0, method="lm", max_nfev=200 * C.FG_GN_ITERATIONS,
            args=(window_nodes, prior_x, flight.accel_meas_avg, flight.baro_meas,
                  detections_by_node, None, dt),
        )
        X = result.x.reshape(len(window_nodes), 9)
        for i, node in enumerate(window_nodes):
            x_est[node] = X[i]
        p_history[k] = X[-1, 0:3]
        # Node `lo` is now finalized (it will be the prior for the next
        # window); keep only what is needed going forward.
        for node in list(x_est.keys()):
            if node < lo:
                del x_est[node]

    return p_history
