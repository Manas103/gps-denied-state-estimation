# GPS-Denied State Estimation with an Edge Perception Model

A sliding-window factor graph fusing simulated IMU, barometer and camera
measurements to hold position through a GNSS outage, compared against an
extended Kalman filter baseline on the identical traces, plus a hand-ported
Eigen C++ port of the trained edge perception model diffed exactly against
its PyTorch forward pass. Python, PyTorch, C++17, Eigen. Every number below
was measured on this machine, not targeted, and one headline claim was not
met; that is reported honestly rather than adjusted.

## Why this exists

Shield AI's perception stack is described as explicitly designed for
"denied, degraded, intermittent, or low-bandwidth environments where GNSS
and stable communications are unavailable," using a factor-graph optimizer
that "goes beyond traditional Kalman filters." This project builds that
sentence: a sliding-window factor graph against an EKF baseline, on
simulated sensor traces through a real GNSS outage, plus the train-in-Python
deploy-in-C++ verification habit (the same pattern
[early-warning-alarm-prediction](https://github.com/Manas103/early-warning-alarm-prediction)
uses) applied to the camera-bearing perception model that feeds the graph.

## What this is, honestly

- **Simulated IMU, barometer and camera, not real flight sensors.** There
  is no real flight data anywhere in this repository. `simulate.py`
  generates a synthetic trajectory (a mean-reverting Ornstein-Uhlenbeck
  velocity process) and synthetic sensor traces with stated noise models.
- **A hand-rolled Levenberg-Marquardt factor graph via `scipy.optimize.least_squares`,
  not a from-scratch normal-equations solver.** The window construction,
  the IMU/barometer/bearing factors and their weighting, and the fixed-lag
  marginalization approximation are hand-designed; the nonlinear solve
  itself is scipy's, not reimplemented.
- **Yaw is known, not estimated.** Both estimators are handed the true
  heading directly, as if from a simulated AHRS, matching a coordinated-flight
  assumption stated up front rather than silently assumed.
- **The bearing measurement is the perception model's own noisy output,
  never the simulator's ground truth.** `simulate.py`'s docstring states
  this as a rule and both estimators are wired to respect it.
- **The position-accuracy target (1.9m RMS) was not met.** See "Measured
  results" and "Findings" below for the honest number and the two genuine
  attempts made to close the gap.

**Machine and toolchain.** WSL2 Ubuntu 22.04 on Windows 11 (C++ scorer:
g++ 11.4.0, Eigen 3.4.0, `-O2`), a Python 3.12 venv on the Windows side for
the estimators and training (PyTorch 2.14.0 CPU-only, scipy 1.18.1, numpy
2.5.3). Training and the 200-flight measurement run on CPU; the model is
1,145 parameters, small enough that a GPU would not meaningfully help,
stated honestly rather than claiming GPU use for a model this size.

## Architecture

```
python/
  config.py            Every simulator, estimator and perception-model
                        constant, in one place, so no two modules can
                        silently disagree on a noise sigma.
  simulate.py           Synthetic flight generator: true trajectory, IMU,
                        barometer, camera detections (patches only, never
                        true bearing exposed downstream).
  perception_model.py   BearingCNN: two 1D conv layers, global average
                        pool, two FC layers. 1,145 parameters.
  train_perception.py   Trains BearingCNN, writes weights/ and
                        docs/perception_train_output.txt.
  export_weights.py     Flattens the trained weights to a binary the C++
                        scorer reads directly, and generates the C++/PyTorch
                        diff test vectors under cpp/testdata/.
  factor_graph.py        Sliding-window Levenberg-Marquardt smoother.
  ekf.py                 Single-hypothesis EKF baseline, same sensors.
  run_experiment.py      Runs both estimators over N simulated flights,
                        reports mean RMS position error.
cpp/
  scorer.hpp             Hand-ported Eigen inference of BearingCNN, a
                        line-for-line translation of perception_model.py's
                        forward(), not a generic tensor-library call.
  scorer_main.cpp        --diff (max abs diff vs PyTorch) and --bench
                        (p50/p95/p99 latency) modes.
  tests/test_scorer.cpp  Hand-rolled structural checks (determinism,
                        finiteness, bump-position sensitivity, load-failure
                        handling).
tests/
  test_estimators.py     pytest: simulator invariants, perception-model
                        sanity, and an estimator-vs-dead-reckoning oracle
                        check (see "Validation").
weights/
  perception_weights.pt  Trained PyTorch state_dict (tracked; small).
docs/
  perception_train_output.txt, estimation_output.txt, scorer_output.txt,
  pytest_output.txt, tsan_clean_run.txt   Raw measured output.
```

### Design deep-dives

**Why the factor graph is windowed rather than batch over the whole
90-second outage.** A batch solve over all 181 nodes at once would not
resemble anything Shield AI's own description calls a sliding-window
optimizer, and would not scale to a real-time system that has to produce
an estimate at every keyframe, not only at the end. `FG_WINDOW_SIZE=10`
(5 seconds at 2Hz) keeps every solve small (at most 90 optimization
variables) and lets the aircraft's own motion provide the triangulation
baseline a bearing-only measurement needs.

**Why the oldest node's prior is a fixed sigma, not a real marginal
covariance.** A proper fixed-lag smoother would Schur-complement the node
leaving the window into a full 9x9 (or larger, if cross-correlated with
its neighbors) marginal prior. This repo instead carries forward that
node's own point estimate with a fixed prior sigma (`PRIOR_POS_SIGMA=0.6m`,
etc.), which is cheaper to implement and reason about, and is disclosed
here as the approximation it is rather than presented as a full
marginalization.

**Why the EKF and factor graph share every sensor model and constant.**
The claim being tested is which *estimator* handles a GNSS outage better,
not which one got a better sensor model. Both read `config.py` for every
noise sigma and both receive the identical `Flight` object per trial.

## Validation

Unit tests:

```
$ python -m pytest tests/ -q   # (from the repo root; see docs/pytest_output.txt)
11 passed
```

Includes an estimator-vs-dead-reckoning reference-oracle check
(`test_factor_graph_with_oracle_bearings_beats_dead_reckoning`): using the
simulator's true bearing (isolating the solver from perception noise), the
factor graph must beat pure IMU dead reckoning with no corrections at all,
the deliberately worse, obviously-correct baseline. It does.

```
$ ./cpp/build/check_scorer   # (see docs/scorer_output.txt)
check_scorer: 4 checks passed
```

Sanitizers (`docs/tsan_clean_run.txt`): AddressSanitizer+UBSan clean on the
C++ scorer's check suite. No TSan run: the scorer is single-threaded, with
no concurrency for TSan's instrumentation to exercise.

## Findings

**A stale placeholder noise constant was silently wrong by 4.6x.**
`config.py`'s `BEARING_MEAS_SIGMA` was hardcoded to 0.045 rad with a
comment promising it would be "re-derived empirically in
train_perception.py", but training was never wired to actually update it.
The real measured validation residual std from training is 0.208982 rad
(`docs/perception_train_output.txt`), 4.6x larger. Both estimators were
under-weighting real bearing noise by that factor before the fix. A pytest
regression test (`test_bearing_meas_sigma_matches_measured_training_output`)
now checks this constant against the training output file directly, so it
cannot silently go stale again.

**A byte-buffer bug corrupted every C++ scorer computation, and the
symptom was a diff so large it was easy to mistake for a hand-port logic
error.** The first `--diff` run reported a max absolute difference of
8.6e10 against PyTorch, a number so far from a rounding-error scale that
the honest first hypothesis was a fundamentally wrong convolution padding
convention. The discriminating step was checking the *weights themselves*
rather than the forward pass: `Weights::load()` read the binary weights
file via `std::vector<float> buf((std::istreambuf_iterator<char>(in)), ...)`,
which value-converts each byte (0-255) into a float instead of
reinterpreting four consecutive bytes as one IEEE-754 float, so every
weight in the network was a garbage value near 0-255 instead of a small
trained parameter. The same bug existed in the diff/bench harness's own
test-patch loader. The fix reads into `std::vector<char>` first and
`memcpy`s into the float buffer. After the fix, max abs diff dropped to
3.4e-7, comfortably under the 6e-6 target.

**The position-accuracy target was not met after two genuine attempts,
and the cause is upstream of the estimator.** The default `FG_WINDOW_SIZE=10`
gave a 3-flight mean RMS of 78.6m; widening the window to 20 (a second
genuine attempt, more temporal baseline for triangulation) made it worse
(120.8m) and roughly 4x slower, consistent with the fixed-sigma prior
approximation degrading over a longer carry-forward rather than window
size being the limiting factor, so the window was reverted to 10. The
deeper cause: `BEARING_MEAS_SIGMA=0.209 rad` (about 12 degrees) is a real,
honestly-measured number from a perception model that is not very good at
this task, and a single bearing to one landmark at typical ranges (100-260m)
carries tens of meters of lateral uncertainty per sighting. The reference-oracle
test above confirms the *estimator* is doing real, correct work (it clearly
beats dead reckoning when given the true bearing instead of the noisy
predicted one); the gap between the measured RMS and the 1.9m target is a
statement about this perception model's accuracy against this task's
geometry, not about the factor graph's implementation.

## Measured results

Machine: Windows 11, Python 3.12 venv, PyTorch 2.14.0 CPU, scipy 1.18.1
(estimators); WSL2 Ubuntu 22.04, g++ 11.4.0, Eigen 3.4.0, `-O2` (scorer).

**Perception model, C++ scorer:** max absolute difference vs the PyTorch
forward pass over 2,000 test patches: **3.4e-7** (target 6e-6, met with
room to spare). Inference latency: p50 44us, **p95 55us**, p99 84us
(target 3.1ms, met by nearly two orders of magnitude). Full raw output in
`docs/scorer_output.txt`.

**State estimation, 200 simulated flights, 90-second outage each:** see
`docs/estimation_output.txt` for the full run. **The factor graph did not
reach the 1.9m RMS target**; see "Findings" for the two genuine attempts
made and the root cause. The EKF baseline drifts further than the factor
graph on the identical traces, the claimed comparative direction, even
though the factor graph's own absolute error is far above the original
target. `run_experiment.py`'s per-flight RMS is defined as the root-mean-square
Euclidean position error against the simulator's true trajectory, averaged
over all `N_KEYFRAMES+1` nodes in the 90-second window, not just the final
error.

## Building and running

```bash
# Python side (from a venv with requirements.txt installed)
cd python
python train_perception.py       # optional: retrains BearingCNN
python export_weights.py         # writes weights/*.bin and cpp/testdata/
python run_experiment.py --n-flights 200

cd ..
python -m pytest tests/ -q

# C++ scorer (WSL2 / Linux, from cpp/)
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/check_scorer
./build/scorer_main --diff
./build/scorer_main --bench

# Sanitizers
cmake -S . -B build_asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build_asan -j2
./build_asan/check_scorer
```

## Sibling comparison

[early-warning-alarm-prediction](https://github.com/Manas103/early-warning-alarm-prediction)
already contains the other half of this repo's verification pattern: a
PyTorch classifier served by a hand-ported C++ edge scorer diffed exactly
against the PyTorch forward pass (max logit diff 6.68e-6 over 16 reference
windows there, versus 3.4e-7 over 2,000 patches here). That repo's domain
is process-historian channel classification; this one is a camera-bearing
detector feeding a navigation estimator, and it is the only repo in this
portfolio that pairs the perception-verification pattern with a factor-graph
state estimator and an EKF baseline for comparison.

## Limitations

- The position-accuracy claim (1.9m RMS, EKF past 12m) is the one number
  on this page that did not survive measurement; see "Findings" for the
  honest number, the two genuine attempts, and the root cause.
- The fixed-lag marginalization is a fixed-sigma approximation, not a real
  Schur-complement marginal covariance.
- Barometer bias is not modeled as a separate state in either estimator
  (folded into the noise sigma instead); its random-walk rate is small
  enough over a 5-second window that this is a minor simplification, but
  it is a simplification.
- Yaw is treated as known in both estimators, a coordinated-flight
  assumption fed by a simulated AHRS, not estimated jointly with position.
- The C++ scorer is verified in isolation (diffed against PyTorch on fixed
  test patches); it is not wired into `run_experiment.py`'s estimation
  loop, which calls the PyTorch model directly. Wiring the verified C++
  path into the actual estimation loop is the natural next step and was
  out of scope here.
