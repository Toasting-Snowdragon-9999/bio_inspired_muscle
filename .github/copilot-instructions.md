# Copilot Instructions — bio_inspired_muscle

## Project Plan

Always read and follow the project plan in `python/README.md` before making changes.
The `python/README.md` describes the architecture, active files, research references, development roadmap, and coding conventions.
Treat `python/README.md` as the single source of truth for what to build and how.
The root `README.md` provides a short project overview.

## Scope

This project's primary codebase lives in the `python/` directory.
Focus assistance on Python code within `python/` and its sub-modules unless explicitly asked otherwise.

## Permissions
You have permission to generate new code, modify existing code, and add comments within the `python/` directory.
When modifying existing code, preserve all existing comments and add new ones for any non-trivial logic.
NEVER RUN THE rm COMMAND without permission. If you think a file is obsolete, flag it in `python/README.md` but do not delete it. E

## Comments Policy

**Never remove, modify, or shorten existing comments in Python files.**
Comments document design decisions, paper references, TODOs, and professor-provided pseudocode.
When adding new code, include clear comments explaining the reasoning — especially when implementing equations from research papers.

## Research Context

The project is grounded in published research. Relevant papers are stored in `Docs/`:

| Directory | Topic | Related module |
|---|---|---|
| `Docs/cpg_related/` | Central Pattern Generators (Matsuoka, Kuramoto oscillators) | `python/cpg/` |
| `Docs/pd_related/` | Adaptive impedance / muscle-like PD control | `python/pd/` |
| `Docs/fuzzy_logic_related/` | Fuzzy logic control (upcoming) | TBD |

When working on a module, consult the corresponding papers for context on equations and parameter choices.

## Technology Stack

- **Python** 3.12.3
- **MuJoCo** for physics simulation (≥ 3.4.0)
- **NumPy**, **SciPy**, **Matplotlib** for numerics and plotting
- **Brian2** for neural oscillator simulations
- **Robot**: Unitree Go2 quadruped (MuJoCo model in `python/sim/go2/`)

## Coding Conventions

- Use **type hints** on function signatures (see `muscle_like_pd.py` for examples).
- Use **dataclasses** for structured parameter groups (see `matsouka_cpg.py` `Neuron`).
- Use **numpy arrays** for vectorised math — avoid Python loops over joints/neurons where possible.
- Enum classes (`Joint`, `Foot`, `Gait`, `State`) are defined in `python/shared_module/robot_state.py` — use them instead of raw strings or ints.
- Constants and lookup dicts live in `python/shared_module/global_constants.py`.
- Virtual environment: `python/.bach_env/` — never commit venv files.
- Files prefixed with `(old)` are legacy and should not be modified or imported.
- Keep imports at the top of files; use the `sys.path.insert` pattern already in use when cross-module imports are needed.

## When Generating or Modifying Code

1. Check the README roadmap to understand what phase the feature belongs to.
2. Reference the relevant paper(s) in `Docs/` for domain correctness.
3. Preserve all existing comments and add new ones for non-trivial logic.
4. Follow the existing code style and conventions listed above.
5. Do not introduce new dependencies without discussion.

## Control Pipeline — Signal Flow

The primary control loop runs each MuJoCo timestep via `IKController.run()`. Understanding this chain is essential for debugging locomotion issues.

### Full chain: CPG → Trajectory → IK → PD → Torque

```
KuramotoCpg.run()
  → RK4 integration of Kuramoto oscillators
  → Outputs: phase θ [4] and phase velocity dθ/dt [4]
      ↓
TrajectoryBuilder.build_trajectory(θ, dθ/dt)
  → Position:  x = -width·cos(θ),  z = amp·sin(θ)   (amp blends height↔stance_depth)
  → Velocity:  dx = width·sin(θ)·dθ/dt,  dz = amp·cos(θ)·dθ/dt
  → Output: per-foot Cartesian targets + velocities in hip frame
      ↓
IKController (per leg)
  → LevenbergMarquardtIK: Cartesian position → joint angles q [3]
  → leg_jacobian(q): 3×3 analytical Jacobian
  → Joint velocities: dq = pinv(J) @ v_cartesian, clamped to ±MAX_JOINT_VEL
      ↓
MuscleLikePD.control(q_target, dq_target)
  → Per leg: e = q_d - q,  de = dq_d - dq
  → ada_imp_ctrl.update_impedance() → K, B matrices (adaptive or fixed)
  → tau = K @ e + B @ de,  clipped to ±ACTUATOR_TORQUE_LIMIT (23.7 Nm)
      ↓
robot_interface.target_torques → MuJoCo actuators
```

### Key parameters and where they live

| Parameter | File | Purpose |
|---|---|---|
| `width` (stride length) | `ik_controller.py` constructor | Horizontal foot excursion |
| `step_height` | `ik_controller.py` constructor | Vertical foot lift per leg |
| `stance_depth` | `trajectory_builder.py` | Ground compliance during stance (0.02 m) |
| `blend_sharpness` | `trajectory_builder.py` | Swing↔stance transition smoothness |
| `MAX_JOINT_VEL` | `ik_controller.py` | Joint velocity clamp (rad/s) |
| `a, b, k` | `adaptive_imp.py` | IOAC adaptation rate, sensitivity, vel weight |
| `kp, kd` | `adaptive_imp.py` (non-adaptive branch) | Fixed PD gains (90, 15) |
| `ACTUATOR_TORQUE_LIMIT` | `global_constants.py` | Motor torque clamp (23.7 Nm) |
| `coupling_strength` | `kuramoto_cpg.py` | Oscillator coupling (1.5π) |
| `frequency` | `robot_state.py` State | CPG base frequency (Hz) |

### Known issues & tuning notes

- **Velocity magnitude scales with CPG frequency.** At 1.5 Hz, dθ/dt ≈ 9.4 rad/s, producing Cartesian velocities up to ~0.9 m/s. Reduce frequency or increase MAX_JOINT_VEL clamp if gait looks sluggish.
- **Adaptive impedance (IOAC)** computes K, B as outer products of tracking errors. Large velocity errors at ground contact can produce aggressive torques. Parameters `a` (numerator, default 0.2) and `b` (denominator sensitivity, default 5.0) control reactivity.
- **Phase velocity** uses the RK4-averaged derivative for consistency with actual phase integration.
- **Abduction joints** (hip lateral) are zeroed in IK — not actively controlled by the sagittal-plane trajectory.
