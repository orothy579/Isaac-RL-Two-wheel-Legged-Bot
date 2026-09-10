# Stair oracle archive

## Human waypoint workbench (start here)

`waypoint_editor.html` is an offline Korean editor. Open it in a browser, import
`waypoints/example.json`, edit times and motor targets, then save your JSON.
The supplied example is an **unvalidated motion seed**, not an expert demonstration.

```bash
# CPU format, timing and commanded speed/torque checks; no simulator needed
python scripts/co_rl/stair_oracle/replay_waypoints.py --profile /absolute/path/my_waypoints.json --validate-only

# Physics replay and 10-second video (10 s is controlled by duration_s in the JSON)
bash scripts/co_rl/stair_oracle/run_waypoints.sh --profile /absolute/path/my_waypoints.json --headless

# Omit --headless for the Isaac Sim window; --no-video skips camera rendering.
```

V1 is a **single finite tread** (15 cm rise, 30 cm depth, 2 m lateral width by default),
with the robot approaching along +x at yaw 0°. It is not the RL pyramid terrain.
The front riser is x=0, the back edge x=0.30, and the ground z=0. `distance_m`
is the initial base-origin perpendicular distance to the front riser, not a
wheel-to-edge distance. The initial base height and initial forward speed are
explicit initial conditions, applied only when the robot is spawned. With the
default initial speed zero, the motor controller must generate the run-up.

Each waypoint contains:

- `t`: absolute simulation time in seconds, strictly increasing from zero.
- `shoulder_rad`: left/right **absolute** shoulder angles in radians.
- `wheel_rad_s`: left/right wheel angular velocity targets, in rad/s.
- `feedforward_nm`: optional extra left shoulder/right shoulder/left wheel/right
  wheel torques, added to the existing PD controller before torque saturation.
- `base_reference`: desired world x, z and pitch [m, m, rad], **comparison only**.
  The controller cannot directly actuate the floating base; it is never moved to
  this reference. V1 is a joint-space editor, not an inverse-kinematics body-pose editor.

The trajectory uses quintic interpolation and zero shoulder target velocity at
each waypoint; it holds the final command through `duration_s`. Reference timing
does not wait for physical arrival: missing a waypoint is logged as tracking error.
The native USD joint limits are checked before playback. Verified hardware limits
are **17 Nm and 40 rad/s**, confirmed by the user, and are set both on the actuator
and the PhysX motor limits. Speed/position excursions are separately reported.
The existing PD gains are retained; replay uses zero delay and no randomized
mass/pushes. The robot USD and its existing collision geometry are retained.

Outputs are isolated in `waypoint_runs/<timestamp>/`:

| File | Meaning |
| --- | --- |
| `report.html` | Offline video/time-slider and selectable signal plot |
| `replay.mp4` | All motion, including failures, with time and motor telemetry |
| `motor_traces.png` | Per-motor requested/applied torque and target/actual tracking |
| `waypoint_checks.csv` | Arrival errors and segment peak torque/saturation duration |
| `trace.csv` | Every 5 ms: motor states, commands, torques, contact and body references |
| `transitions.npz` | Pre-step state, command, post-step state, command timestamp |
| `profile.json` | Exact input profile |
| `summary.json` | Provenance, physics/control settings, limits and landing result |

`torque_requested` is the torque demanded by this PD + feedforward controller;
`torque_applied` is the bounded explicit actuator output passed to PhysX. Neither
is a unique inverse-dynamics solution or proof of the minimum torque needed for
the intended body trajectory. The workbench tests **whether this command sequence
can track the waypoints**, and where it saturates or misses. It does not solve
trajectory optimization automatically. Electrical current is not fabricated from
torque: a torque constant/gearing/efficiency model would be needed for that conversion.

Jump success requires a supported robot to enter a >=30 ms flight phase, then end
with both wheels contacting within the tread, acceptable tilt/body velocity, and
at least 0.5 s continuous final stable support, with motor-limit checks passed.
Every replay starts with `bc_expert_approved=false`, even if that physics predicate
passes. These state/command sequences are **candidate demonstrations**, not BC
training and not directly compatible with the existing policy observation/action
format. That conversion and BC training are a later step after inspecting a valid motion.

CPU tests:

```bash
python -m unittest scripts.co_rl.stair_oracle.test_waypoints -v
```

### Resource limits and shutdown

Always use `run_waypoints.sh` for GPU replay. It starts a separate systemd user
scope (outside the VS Code scope) with MemoryHigh=5 GiB, MemoryMax=6 GiB,
MemorySwapMax=256 MiB, and a 600 s wall-time limit (TERM, then KILL after 10 s).
If the budget is insufficient, the replay fails rather than silently removing
the cap. The Python entry point also takes a process-lifetime exclusive lock,
and rejects startup when less than 6 GiB RAM is available. The lock stays held
through shutdown. `--validate-only` runs without Isaac Sim.

The local Isaac Lab/Kit native shutdown path can retain the Python GIL while
hanging. This dedicated CLI therefore closes every result writer, unregisters
simulation callbacks, records `shutdown.json`, and exits the process directly;
the OS then reclaims Kit and GPU resources. Check process exit, not just a saved
report, before launching another simulation. Direct invocation of the Python GPU
entry point does not create the memory-limited systemd scope.

See `INCIDENT_2026-09-05.md` for the OOM investigation and verified facts.

Verified rendered example: `waypoint_runs/2026-09-05_09-39-42_554503/report.html`.
10 s / 200 video frames / 2,000 physics samples; the process exited with code 0
and the scope became inactive. This example failed the jump and is not BC expert
data. The maximum shoulder requests were 21.94/22.57 Nm, bounded to 17 Nm.

## Earlier RL diagnostics (different data source)

This directory keeps diagnostic data and short visual evidence separate from the RL logs:

- `data/<training-run>/stair_attempts.csv`: one row per sampled jump attempt.
- `data/<training-run>/stair_attempts_metadata.json`: provenance and success-label definition.
- `data/<training-run>/stair_capability_map.csv`: binned distance/angle/speed/torque results.
- `videos/<video-eval-run>/*.mp4`: only oracle-labelled successes, 10 seconds by default.
- `videos/<video-eval-run>/*.json`: checkpoint, stair dimensions, distance, angle, and landing result for the matching clip.

The oracle values are logging/evaluation signals only. They are not policy observations and do not alter rewards, actions, commands, or terminations.

## Build a capability map

```bash
python scripts/co_rl/analyze_stair_attempts.py \
  scripts/co_rl/stair_oracle/data/<training-run>/stair_attempts.csv
```

## Record successful replays

Run after selecting a checkpoint. Level 9 is the final 15 cm height / 30 cm tread target.

```bash
python scripts/co_rl/stair_oracle/record_successes.py \
  --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo \
  --checkpoint /absolute/path/to/model_5000.pt \
  --terrain-level 9 --terrain-type 5 --command-vx 0.7 \
  --num-clips 5 --clip-seconds 10 \
  --headless
```

The recorder searches with one rendered environment, buffers the approach, and encodes a clip only if the same attempt satisfies the oracle success definition: sufficient climb, both wheels in contact, and stable final roll/pitch. By default it retains the checkpoint's small learned action-noise distribution so it represents the successful training rollouts; add `--deterministic` to evaluate mean actions only.
