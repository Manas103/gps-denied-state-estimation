"""Shared constants for the simulation, estimators and perception model.

All units SI (meters, seconds, radians) unless noted. Every sigma below is a
1-sigma standard deviation. These are simulator design choices, not measured
facts; they are stated here once so every module (and the README) references
the same numbers.
"""

import numpy as np

# ---- Timing ----
OUTAGE_DURATION_S = 90.0          # the claimed GNSS outage window
IMU_SUBSAMPLE_HZ = 20.0           # raw simulated IMU sub-sample rate
KEYFRAME_HZ = 2.0                 # sliding-window node / baro / camera rate
DT_KEYFRAME = 1.0 / KEYFRAME_HZ
N_SUB_PER_KEYFRAME = int(IMU_SUBSAMPLE_HZ * DT_KEYFRAME)
N_KEYFRAMES = int(OUTAGE_DURATION_S * KEYFRAME_HZ)

# ---- Flight profile (world frame, x east-ish, y north-ish, z up) ----
# The true trajectory is generated from a mean-reverting (Ornstein-Uhlenbeck)
# velocity process, not a mean-reverting acceleration process: integrating a
# random acceleration directly would let position wander with variance that
# grows as t^3 (unrealistically large excursions over 90 s). A mean-reverting
# velocity keeps the aircraft on a bounded, gently wandering track, and the
# simulated accelerometer measures its time-derivative, exactly as a real
# accelerometer would.
CRUISE_SPEED_MEAN = 18.0          # m/s forward speed, roughly a small fixed-wing UAV
CRUISE_ALT_MEAN = 150.0           # m, nominal altitude (z, positive up)
VX_STD = 1.5                      # m/s, forward speed wander
VY_STD = 2.0                      # m/s, lateral speed wander (gentle S-turns)
VZ_STD = 0.5                      # m/s, vertical speed wander (gentle climbs/descents)
VX_TAU = 8.0                      # s, OU time constant, forward
VY_TAU = 5.0                      # s, OU time constant, lateral
VZ_TAU = 8.0                      # s, OU time constant, vertical

# ---- IMU (accelerometer) error model ----
ACCEL_WHITE_NOISE_SIGMA = 0.06        # m/s^2, per raw sub-sample, per axis
ACCEL_BIAS_RW_SIGMA = 0.004           # m/s^2 / sqrt(s), random-walk rate, per axis
ACCEL_BIAS_INIT_SIGMA = 0.05          # m/s^2, bias uncertainty at outage start

# Effective noise of the per-keyframe *averaged* accel measurement (an
# average of N_SUB_PER_KEYFRAME independent white-noise sub-samples has
# sigma/sqrt(N) noise), and what a one-step IMU integration of that noise
# does to velocity and position. Both estimators (factor_graph.py, ekf.py)
# use these, so they are derived once, here, rather than duplicated.
ACCEL_MEAS_AVG_SIGMA = ACCEL_WHITE_NOISE_SIGMA / np.sqrt(N_SUB_PER_KEYFRAME)
ACCEL_WHITE_NOISE_SIGMA_V_EQUIV = ACCEL_MEAS_AVG_SIGMA * DT_KEYFRAME
ACCEL_WHITE_NOISE_SIGMA_P_EQUIV = 0.5 * ACCEL_MEAS_AVG_SIGMA * DT_KEYFRAME ** 2

# ---- Barometer error model ----
BARO_NOISE_SIGMA = 0.35               # m, white noise per measurement
BARO_BIAS_RW_SIGMA = 0.01             # m / sqrt(s), slow drift
BARO_BIAS_INIT_SIGMA = 0.5            # m

# ---- Initial state uncertainty at outage start (identical for FG and EKF) ----
POS_INIT_SIGMA = 0.3                  # m, per axis
VEL_INIT_SIGMA = 0.05                 # m/s, per axis

# ---- Camera / landmark perception ----
FOV_HALF_ANGLE_RAD = 0.6              # +/- ~34 deg field of view
MAX_LANDMARK_RANGE_M = 260.0          # beyond this the landmark is too small to detect
CAMERA_DETECTION_DROPOUT_P = 0.08     # probability a visible landmark is missed
CAMERA_PATCH_LEN = 64                 # 1D synthetic "scanline" length fed to the CNN
CAMERA_BUMP_SIGMA_PX_RANGE = (1.5, 3.0)
CAMERA_BUMP_AMPLITUDE_RANGE = (0.6, 1.0)
CAMERA_PIXEL_NOISE_SIGMA = 0.05
BEARING_MEAS_SIGMA = 0.208982         # rad, effective sigma of the CNN bearing output.
                                       # This is the measured val_resid_std_rad from
                                       # train_perception.py (docs/perception_train_output.txt),
                                       # not a guess: an earlier placeholder of 0.045 rad here
                                       # was never updated after training actually ran, which
                                       # would have under-weighted the real measurement noise
                                       # in both estimators by roughly 4.6x. See README "Findings".

# World-frame landmark positions (a small mapped visual-landmark database).
# Placed ahead of and around the nominal cruise track so different flights
# see different subsets depending on their randomized initial heading/offset.
LANDMARKS = np.array([
    [250.0,   40.0,  30.0],
    [520.0,  -60.0,  20.0],
    [800.0,   30.0,  45.0],
    [1080.0, -40.0,  25.0],
    [1350.0,  50.0,  35.0],
], dtype=np.float64)

# ---- Sliding window factor graph ----
FG_WINDOW_SIZE = 10                   # keyframes kept in the active window (5 s at 2 Hz).
                                       # A 20-keyframe window was also tried (a genuine second
                                       # attempt at the RMS target) and made the estimate worse
                                       # (120.8m vs 78.6m mean RMS on a 3-flight check) as well
                                       # as ~4x slower, consistent with the fixed-sigma prior
                                       # approximation degrading over a longer carry-forward
                                       # rather than the window being the limiting factor. See
                                       # README "Findings".
FG_GN_ITERATIONS = 4                   # Gauss-Newton iterations per incremental solve

# ---- Misc ----
GRAVITY = 9.81  # informational only; the simulated IMU below outputs specific
                 # force already resolved to world frame net of gravity (see
                 # README "what this is NOT"), so GRAVITY is not subtracted
                 # anywhere in this codebase.

RNG_SEED_BASE = 20260911  # deterministic per-flight seeding: seed = BASE + flight_id
