"""Synthetic flight generator for the GNSS-outage state estimation study.

Everything in this module is SIMULATED. There is no real IMU, barometer or
camera data anywhere in this repository. The generator produces, per flight:

  - a true trajectory (an Ornstein-Uhlenbeck-driven random acceleration,
    forward-integrated with simple Euler steps at IMU_SUBSAMPLE_HZ),
  - a simulated accelerometer trace with additive white noise and a
    random-walk bias (the classic MEMS IMU error model), pre-averaged into
    one "preintegrated" measurement per keyframe interval,
  - a simulated barometer trace (altitude plus white noise plus a slow
    random-walk bias),
  - simulated camera detections of a small fixed set of world-frame
    landmarks: a landmark is "detected" only if it falls inside the
    field of view and within range, and detections are further dropped
    with a fixed miss probability. Each detection carries a synthetic
    1D pixel "scanline" patch (see perception.py) whose bump position
    encodes the true bearing angle plus sensor noise; nothing downstream
    is allowed to read the true bearing directly, only the patch.

Attitude (yaw) is not estimated: it is derived directly from the true
velocity heading and handed to the estimators as a known quantity, exactly
like a coordinated-flight assumption fed by a simulated AHRS. This is a
disclosed simplification, see the README "what this is NOT" section.
"""

from dataclasses import dataclass, field

import numpy as np

import config as C


def _ou_process(n, dt, tau, sigma, rng):
    """Discrete Ornstein-Uhlenbeck process, stationary std = sigma."""
    a = np.zeros(n)
    decay = np.exp(-dt / tau)
    diffusion = sigma * np.sqrt(1.0 - decay ** 2)
    for i in range(1, n):
        a[i] = decay * a[i - 1] + diffusion * rng.standard_normal()
    return a


def _wrap_angle(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


@dataclass
class CameraDetection:
    keyframe: int
    landmark_idx: int
    true_bearing: float
    true_range: float
    patch: np.ndarray


@dataclass
class Flight:
    seed: int
    n_keyframes: int
    dt_keyframe: float
    p_true: np.ndarray          # (n_keyframes+1, 3), position at each node time
    v_true: np.ndarray          # (n_keyframes+1, 3)
    ba_true: np.ndarray         # (n_keyframes+1, 3), accel bias at each node time
    accel_meas_avg: np.ndarray  # (n_keyframes, 3), averaged over each interval
    baro_meas: np.ndarray       # (n_keyframes+1,)
    detections: list = field(default_factory=list)  # list[CameraDetection]
    p0_est_error: np.ndarray = None   # initial position error handed to both estimators
    v0_est_error: np.ndarray = None
    ba0_est_error: np.ndarray = None


def make_camera_patch(true_bearing, rng):
    """Render a synthetic 1D camera scanline encoding true_bearing as a bump."""
    u = np.clip(true_bearing / C.FOV_HALF_ANGLE_RAD, -1.0, 1.0)
    center_px = (u * 0.5 + 0.5) * (C.CAMERA_PATCH_LEN - 1)
    sigma_px = rng.uniform(*C.CAMERA_BUMP_SIGMA_PX_RANGE)
    amp = rng.uniform(*C.CAMERA_BUMP_AMPLITUDE_RANGE)
    x = np.arange(C.CAMERA_PATCH_LEN, dtype=np.float64)
    patch = amp * np.exp(-0.5 * ((x - center_px) / sigma_px) ** 2)
    patch += rng.normal(0.0, C.CAMERA_PIXEL_NOISE_SIGMA, size=C.CAMERA_PATCH_LEN)
    return patch.astype(np.float32)


def generate_flight(flight_id: int) -> Flight:
    seed = C.RNG_SEED_BASE + flight_id
    rng = np.random.default_rng(seed)

    n_sub = C.N_SUB_PER_KEYFRAME * C.N_KEYFRAMES
    dt_sub = 1.0 / C.IMU_SUBSAMPLE_HZ

    # True velocity process, world-frame: each axis is a mean-reverting
    # (Ornstein-Uhlenbeck) process around its cruise value. This keeps the
    # trajectory a bounded, gently wandering flight rather than an unbounded
    # random walk. True acceleration is the finite-difference derivative of
    # this velocity, exactly what a real accelerometer would sense.
    vx = C.CRUISE_SPEED_MEAN + _ou_process(n_sub + 1, dt_sub, C.VX_TAU, C.VX_STD, rng)
    vy = _ou_process(n_sub + 1, dt_sub, C.VY_TAU, C.VY_STD, rng)
    vz = _ou_process(n_sub + 1, dt_sub, C.VZ_TAU, C.VZ_STD, rng)
    v = np.stack([vx, vy, vz], axis=1)  # (n_sub+1, 3)

    a_true = np.zeros_like(v)
    a_true[:-1] = (v[1:] - v[:-1]) / dt_sub
    a_true[-1] = a_true[-2]

    p = np.zeros((n_sub + 1, 3))
    p[0] = np.array([0.0, rng.uniform(-20.0, 20.0), C.CRUISE_ALT_MEAN + rng.uniform(-20.0, 20.0)])
    for i in range(n_sub):
        p[i + 1] = p[i] + v[i] * dt_sub + 0.5 * a_true[i] * dt_sub ** 2

    yaw = np.arctan2(v[:, 1], v[:, 0])

    # Simulated accelerometer: random-walk bias + white noise, at sub-sample rate.
    bias = np.zeros((n_sub + 1, 3))
    bias[0] = rng.normal(0.0, C.ACCEL_BIAS_INIT_SIGMA, size=3)
    rw_step = C.ACCEL_BIAS_RW_SIGMA * np.sqrt(dt_sub)
    for i in range(n_sub):
        bias[i + 1] = bias[i] + rng.normal(0.0, rw_step, size=3)
    accel_meas = a_true + bias + rng.normal(0.0, C.ACCEL_WHITE_NOISE_SIGMA, size=(n_sub + 1, 3))

    n_sub_pk = C.N_SUB_PER_KEYFRAME
    accel_meas_avg = np.array([
        accel_meas[k * n_sub_pk:(k + 1) * n_sub_pk].mean(axis=0)
        for k in range(C.N_KEYFRAMES)
    ])

    node_idx = np.arange(0, C.N_KEYFRAMES + 1) * n_sub_pk
    p_true_nodes = p[node_idx]
    v_true_nodes = v[node_idx]
    ba_true_nodes = bias[node_idx]
    yaw_nodes = yaw[node_idx]

    # Simulated barometer.
    n_nodes = C.N_KEYFRAMES + 1
    baro_bias = np.zeros(n_nodes)
    baro_bias[0] = rng.normal(0.0, C.BARO_BIAS_INIT_SIGMA)
    rw_step_b = C.BARO_BIAS_RW_SIGMA * np.sqrt(C.DT_KEYFRAME)
    for k in range(n_nodes - 1):
        baro_bias[k + 1] = baro_bias[k] + rng.normal(0.0, rw_step_b)
    baro_meas = p_true_nodes[:, 2] + baro_bias + rng.normal(0.0, C.BARO_NOISE_SIGMA, size=n_nodes)

    # Simulated camera detections.
    detections = []
    for k in range(n_nodes):
        px, py, pz = p_true_nodes[k]
        best = None
        for li, L in enumerate(C.LANDMARKS):
            dx, dy = L[0] - px, L[1] - py
            rng_h = np.hypot(dx, dy)
            rng_full = np.sqrt(dx ** 2 + dy ** 2 + (L[2] - pz) ** 2)
            bearing = _wrap_angle(np.arctan2(dy, dx) - yaw_nodes[k])
            if rng_full < C.MAX_LANDMARK_RANGE_M and abs(bearing) < C.FOV_HALF_ANGLE_RAD:
                if best is None or rng_full < best[0]:
                    best = (rng_full, li, bearing)
        if best is not None:
            if rng.random() >= C.CAMERA_DETECTION_DROPOUT_P:
                _, li, bearing = best
                patch = make_camera_patch(bearing, rng)
                detections.append(CameraDetection(
                    keyframe=k, landmark_idx=li, true_bearing=bearing,
                    true_range=best[0], patch=patch,
                ))

    flight = Flight(
        seed=seed,
        n_keyframes=C.N_KEYFRAMES,
        dt_keyframe=C.DT_KEYFRAME,
        p_true=p_true_nodes,
        v_true=v_true_nodes,
        ba_true=ba_true_nodes,
        accel_meas_avg=accel_meas_avg,
        baro_meas=baro_meas,
        detections=detections,
    )
    flight.p0_est_error = rng.normal(0.0, C.POS_INIT_SIGMA, size=3)
    flight.v0_est_error = rng.normal(0.0, C.VEL_INIT_SIGMA, size=3)
    flight.ba0_est_error = rng.normal(0.0, C.ACCEL_BIAS_INIT_SIGMA, size=3)
    flight.yaw_nodes = yaw_nodes
    return flight
