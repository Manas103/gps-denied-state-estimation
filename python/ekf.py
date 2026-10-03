"""Extended Kalman Filter baseline over the identical simulated traces the
factor graph uses. Same 9-state [p, v, ba], same IMU propagation, same
barometer and bearing measurements (the bearing measurement is the trained
perception model's output, exactly as for the factor graph, so the
comparison isolates the estimator, not the sensor).

This is a standard single-hypothesis EKF, not the factor graph's windowed
batch re-optimization: every measurement is fused once, in arrival order,
and never revisited. That is the entire point of the comparison (see
README): a global outage the EKF cannot "look back" and fix, so its
position error accumulates without bound as sensor noise integrates
through the propagation step, and the factor graph's ability to keep the
last FG_WINDOW_SIZE nodes jointly consistent should measurably beat it.
"""
import numpy as np

import config as C


def _wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def run_ekf(flight, bearing_meas_by_det):
    n_nodes = flight.n_keyframes + 1
    dt = flight.dt_keyframe

    detections_by_node = {}
    for det in flight.detections:
        detections_by_node.setdefault(det.keyframe, []).append(det)

    x = np.concatenate([
        flight.p_true[0] + flight.p0_est_error,
        flight.v_true[0] + flight.v0_est_error,
        flight.ba_true[0] + flight.ba0_est_error,
    ])
    P = np.diag([C.POS_INIT_SIGMA ** 2] * 3 + [C.VEL_INIT_SIGMA ** 2] * 3 +
                [C.ACCEL_BIAS_INIT_SIGMA ** 2] * 3)

    Q = np.diag(
        [C.ACCEL_WHITE_NOISE_SIGMA_P_EQUIV ** 2] * 3 +
        [C.ACCEL_WHITE_NOISE_SIGMA_V_EQUIV ** 2] * 3 +
        [(C.ACCEL_BIAS_RW_SIGMA * np.sqrt(dt)) ** 2] * 3
    )
    R_baro = C.BARO_NOISE_SIGMA ** 2
    R_bearing = C.BEARING_MEAS_SIGMA ** 2

    p_history = np.zeros((n_nodes, 3))
    p_history[0] = x[0:3]

    I9 = np.eye(9)
    for k in range(1, n_nodes):
        # --- Propagate ---
        p, v, ba = x[0:3], x[3:6], x[6:9]
        a = flight.accel_meas_avg[k - 1] - ba
        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt
        ba_new = ba.copy()
        x = np.concatenate([p_new, v_new, ba_new])

        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * dt * dt * np.eye(3)
        F[3:6, 6:9] = -dt * np.eye(3)
        P = F @ P @ F.T + Q

        # --- Update: barometer ---
        z = flight.baro_meas[k]
        H = np.zeros((1, 9)); H[0, 2] = 1.0
        y = z - x[2]
        S = H @ P @ H.T + R_baro
        K = P @ H.T / S
        x = x + (K.flatten() * y)
        P = (I9 - K @ H) @ P

        # --- Update: bearing detections at this node ---
        for det in detections_by_node.get(k, []):
            L = C.LANDMARKS[det.landmark_idx]
            dx, dy = L[0] - x[0], L[1] - x[1]
            r2 = dx * dx + dy * dy
            bearing_pred = _wrap(np.arctan2(dy, dx) - flight.yaw_nodes[k])
            z_meas = bearing_meas_by_det[(det.keyframe, det.landmark_idx)]
            y = _wrap(z_meas - bearing_pred)
            H = np.zeros((1, 9))
            H[0, 0] = dy / r2
            H[0, 1] = -dx / r2
            S = H @ P @ H.T + R_bearing
            K = P @ H.T / S
            x = x + (K.flatten() * y)
            P = (I9 - K @ H) @ P

        p_history[k] = x[0:3]

    return p_history
