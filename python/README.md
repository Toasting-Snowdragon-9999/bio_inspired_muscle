# Python — Bio-Inspired Muscle Control

Detailed project plan for the Python codebase. **Copilot: always consult this file before making changes.**

---

## Architecture

```
python/
├── cpg/                  # Central Pattern Generators (locomotion rhythm) Generates a generic output
├── pd/                   # PD / adaptive impedance controllers - Provided by proffesor and should reference Docs/pd_related/ioac_paper.pdf section 3. Online impedance adaptation control.
├── controllers/          # High-level controllers (IK wrapper, etc.) - Uses cpg, trajectory builder, inv kinematics and pd to send torque to the robot
├── inverse_kinematics/   # Leg inverse kinematics solver
├── sim/                  # MuJoCo simulation & test scenes
│   └── go2/              # Unitree Go2 MJCF model and scenes
├── shared_module/        # Enums, constants, robot state interface
├── plotting/             # Time-series visualisation utilities
└── requirements.txt      # Python dependencies
```

---

## Active Files

Only the files listed below are current. Files prefixed with `(old)` are legacy and should **not** be modified or imported.

### cpg/
| File | Description |
|---|---|
| `matsouka_cpg.py` | Matsuoka half-centre oscillator CPG, primary cpg in use.|
| `trajectory_builder.py` | Converts CPG phase output to Cartesian foot trajectories |

### pd/
| File | Description |
|---|---|
| `muscle_like_pd.py` | Muscle-like PD controller with per-leg adaptive impedance |
| `adaptive_imp.py` | Online impedance adaptation (professor-provided, needs implementation & testing) |

### controllers/
| File | Description |
|---|---|
| `ik_controller.py` | The primary controller implementing the entire pipeline from traj building to final torque and uses the robot_interface to send to the robot |

### inverse_kinematics/
| File | Description |
|---|---|
| `inverse_kin.py` | Analytical IK solver for Go2 legs |

### sim/
| File | Description |
|---|---|
| `mujoco_sim.py` | Core MuJoCo simulation loop and rendering |
| `mujoco_kuramoto_test.py` | Kuramoto CPG + IK + PD integration test |
| `IK_test.py` | Inverse kinematics test scene |
| `test_inv_sim.py` | IK integration test |
| `mujoco_manual_control.py` | Manual joint control for debugging |
| `mujoco_test.py` | Basic simulation test |

### shared_module/
| File | Description |
|---|---|
| `robot_state.py` | `RobotInterface`, `Joint`, `Foot`, `Gait`, `State` enums & dataclasses, very important since it is the backbone af intercommunication between classes and simulation|
| `global_constants.py` | Joint ranges, actuator/sensor dicts, gait frequencies |

### plotting/
| File | Description |
|---|---|
| `time_series_plotter.py` | Reusable matplotlib time-series plotter NOT IN USE!|

---

## Research References

Papers in `Docs/` provide the theoretical foundation. Consult the relevant papers when working on a module.

| Docs directory | Topic | Related module |
|---|---|---|
| `Docs/cpg_related/` | Matsuoka oscillators, Kuramoto coupling | `cpg/` |
| `Docs/pd_related/` | Adaptive impedance, muscle-like PD control | `pd/` |
| `Docs/fuzzy_logic_related/` | Fuzzy logic control (upcoming) | TBD |

Key papers:
- **Matsuoka (1985)** — half-centre oscillator model → `cpg/matsouka_cpg.py`
- **Hexel (2015)** — eight-neuron CPG network → `Docs/cpg_related/eight_neuron/`
- **Xiong & Fang (2023)** — online impedance adaptation → `pd/adaptive_imp.py`
- **Learning Control with Emulated Muscle** → `Docs/pd_related/`

---

## Development Roadmap

| Phase | Module | Status |
|---|---|---|
| CPG locomotion rhythm | `cpg/` | ✅ Mostly complete |
| Trajectory generation | `cpg/trajectory_builder.py` | ✅ Mostly complete |
| Inverse kinematics | `inverse_kinematics/` | ✅ Mostly complete |
| Adaptive PD control | `pd/` | 🔧 Implementing & testing |
| Fuzzy logic control | TBD | 📋 Planned |

### Current priorities
1. Integrate and test the adaptive impedance controller (`adaptive_imp.py`) from professor
2. Experiment with Cycloidal Trajectories using the trajectory builder and making a new function like build_trajectory, but building cycloidal.
3. Tune `muscle_like_pd.py` gains with the adaptive impedance backend
4. Validate full pipeline: CPG → trajectory → IK → PD → MuJoCo

### Upcoming
- Fuzzy logic layer for gait transition and terrain adaptation (papers to be added to `Docs/fuzzy_logic_related/`)

---

## Coding Conventions

- **Python 3.12.3** — virtual env in `.bach_env/`
- **Type hints** on all function signatures
- **Dataclasses** for structured parameter groups
- **NumPy arrays** for vectorised math — avoid Python loops over joints/neurons
- Use the **enums** from `shared_module/robot_state.py` (`Joint`, `Foot`, `Gait`, `State`) — no raw strings/ints
- Constants and lookup dicts in `shared_module/global_constants.py`
- **Never remove or modify existing comments** — they document design decisions, paper references, and TODOs
- Files prefixed with `(old)` are legacy — do not modify or import them
- Cross-module imports use the `sys.path.insert` pattern already in place
