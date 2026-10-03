"""pytest suite for the simulator, the perception model integration, and
both estimators. Run from the tests/ directory (adds ../python to sys.path)
so it exercises the real modules under python/, not a copy.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import config as C
from ekf import run_ekf
from factor_graph import run_factor_graph
from perception_model import BearingCNN, load_model
from simulate import generate_flight, make_camera_patch


WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "perception_weights.pt")


# ---------------------------------------------------------------------
# simulate.py
# ---------------------------------------------------------------------

def test_flight_shapes_are_consistent():
    flight = generate_flight(0)
    n_nodes = flight.n_keyframes + 1
    assert flight.p_true.shape == (n_nodes, 3)
    assert flight.v_true.shape == (n_nodes, 3)
    assert flight.ba_true.shape == (n_nodes, 3)
    assert flight.accel_meas_avg.shape == (flight.n_keyframes, 3)
    assert flight.baro_meas.shape == (n_nodes,)


def test_flight_generation_is_deterministic_per_seed():
    a = generate_flight(5)
    b = generate_flight(5)
    assert np.array_equal(a.p_true, b.p_true)
    assert np.array_equal(a.accel_meas_avg, b.accel_meas_avg)


def test_different_flight_ids_diverge():
    a = generate_flight(0)
    b = generate_flight(1)
    assert not np.array_equal(a.p_true, b.p_true)


def test_camera_detections_stay_within_field_of_view():
    flight = generate_flight(0)
    assert len(flight.detections) > 0, "the default flight profile should see at least one landmark"
    for det in flight.detections:
        assert abs(det.true_bearing) <= C.FOV_HALF_ANGLE_RAD
        assert det.true_range <= C.MAX_LANDMARK_RANGE_M


def test_camera_patch_bump_center_tracks_bearing():
    # A patch built for a bearing near +FOV_HALF_ANGLE_RAD should have its
    # energy concentrated near the end of the patch array, and a patch for
    # a bearing near -FOV_HALF_ANGLE_RAD near the start: the encoding the
    # perception model is trained to invert.
    rng = np.random.default_rng(1)
    right = make_camera_patch(C.FOV_HALF_ANGLE_RAD * 0.9, rng)
    left = make_camera_patch(-C.FOV_HALF_ANGLE_RAD * 0.9, rng)
    assert np.argmax(right) > np.argmax(left)


# ---------------------------------------------------------------------
# perception_model.py
# ---------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="trained weights not present")
def test_trained_model_beats_a_random_baseline():
    model = load_model(WEIGHTS_PATH)
    rng = np.random.default_rng(42)
    bearings = rng.uniform(-C.FOV_HALF_ANGLE_RAD, C.FOV_HALF_ANGLE_RAD, size=500)
    patches = np.stack([make_camera_patch(b, rng) for b in bearings]).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(patches).unsqueeze(1)).squeeze(1).numpy()
    mae_rad = np.mean(np.abs(pred * C.FOV_HALF_ANGLE_RAD - bearings))
    # A model that learned nothing would have MAE around FOV_HALF_ANGLE_RAD/2
    # (mean absolute deviation of a uniform distribution from its own mean);
    # the trained model's measured val_resid_std_rad is 0.209, well under that.
    assert mae_rad < C.FOV_HALF_ANGLE_RAD / 2.0


def test_untrained_model_produces_finite_output():
    model = BearingCNN()
    model.eval()
    x = torch.zeros(4, 1, C.CAMERA_PATCH_LEN)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()
    assert out.shape == (4, 1)


# ---------------------------------------------------------------------
# Estimators: sanity and invariant checks that do not depend on the full
#200-flight measurement (that is a separate, slower measurement script,
# python/run_experiment.py, whose own raw output is committed under docs/).
# ---------------------------------------------------------------------

def _truth_bearings(flight):
    return {(d.keyframe, d.landmark_idx): d.true_bearing for d in flight.detections}


@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="trained weights not present")
def test_factor_graph_runs_and_returns_full_trajectory():
    flight = generate_flight(0)
    bmeas = _truth_bearings(flight)  # oracle bearings, isolates the solver from perception noise
    p_est = run_factor_graph(flight, bmeas)
    assert p_est.shape == flight.p_true.shape
    assert np.all(np.isfinite(p_est))


@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="trained weights not present")
def test_ekf_runs_and_returns_full_trajectory():
    flight = generate_flight(0)
    bmeas = _truth_bearings(flight)
    p_est = run_ekf(flight, bmeas)
    assert p_est.shape == flight.p_true.shape
    assert np.all(np.isfinite(p_est))


@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="trained weights not present")
def test_factor_graph_with_oracle_bearings_beats_dead_reckoning():
    # Reference oracle: pure IMU dead reckoning, no corrections at all (the
    # deliberately worse, obviously-correct baseline every other repo in
    # this portfolio uses). A working estimator using the *true* bearing
    # (no perception noise) must beat pure dead reckoning; if it does not,
    # the factor graph's measurement updates are not doing anything.
    flight = generate_flight(2)
    bmeas = _truth_bearings(flight)
    p_fg = run_factor_graph(flight, bmeas)

    dt = flight.dt_keyframe
    p_dr = np.zeros_like(flight.p_true)
    p_dr[0] = flight.p_true[0] + flight.p0_est_error
    v = flight.v_true[0] + flight.v0_est_error
    ba = flight.ba0_est_error  # dead reckoning never corrects the bias at all
    for k in range(1, flight.n_keyframes + 1):
        a = flight.accel_meas_avg[k - 1] - ba
        p_dr[k] = p_dr[k - 1] + v * dt + 0.5 * a * dt * dt
        v = v + a * dt

    rms_fg = np.sqrt(np.mean(np.sum((p_fg - flight.p_true) ** 2, axis=1)))
    rms_dr = np.sqrt(np.mean(np.sum((p_dr - flight.p_true) ** 2, axis=1)))
    assert rms_fg < rms_dr


def test_bearing_meas_sigma_matches_measured_training_output():
    # Regression test for the bug fixed this session: BEARING_MEAS_SIGMA
    # must track the actually-measured val_resid_std_rad from
    # docs/perception_train_output.txt, not a stale placeholder.
    train_output_path = os.path.join(os.path.dirname(__file__), "..", "docs",
                                      "perception_train_output.txt")
    if not os.path.exists(train_output_path):
        pytest.skip("training output not present")
    measured = None
    with open(train_output_path) as f:
        for line in f:
            if line.startswith("val_resid_std_rad="):
                measured = float(line.strip().split("=")[1])
    assert measured is not None
    assert C.BEARING_MEAS_SIGMA == pytest.approx(measured, abs=1e-6)
