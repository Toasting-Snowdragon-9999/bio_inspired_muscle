# bio_inspired_muscle

**Bio-inspired muscle control for quadruped locomotion** — a robotics bachelor
project that makes a [Unitree Go2](https://www.unitree.com/go2) quadruped walk,
trot, amble, canter and gallop in the [MuJoCo](https://mujoco.org) physics
simulator. Locomotion is driven by a **Central Pattern Generator (CPG)** of
coupled oscillators, foot trajectories are solved with **inverse kinematics**,
and the joints are actuated by a **muscle-like adaptive-impedance controller
(OIAC)**. Gait parameters are tuned automatically with **reinforcement learning
(PPO / CMA-ES)**.

## Demo

The clip below is the project's result — the Go2 locomoting in MuJoCo:

<video src="https://github.com/Toasting-Snowdragon-9999/bio_inspired_muscle/raw/main/Docs/Quadruped_locomotion.mp4" controls muted loop width="100%"></video>

> ▶️ If the player above does not appear in your viewer, watch / download it here:
> [`Docs/Quadruped_locomotion.mp4`](Docs/Quadruped_locomotion.mp4)

---

## Architecture

The controller is a feed-forward pipeline driven by a shared state object
(`RobotInterface`). Each control step flows top-to-bottom; sensor readings flow
back into the shared state:

```
                ┌─────────────────────────────────────────────┐
                │  RobotInterface  (central shared state hub)   │
                │  gait, frequency, joint pos/vel, contacts …   │
                └───────────────┬───────────────────▲───────────┘
                                │                   │ sensors
                                ▼                   │
   KuramotoCpg ──► TrajectoryBuilder ──► Levenberg-  ──► MuscleLikePD ──► MujocoSim
   (phase per     + GaitScheduler        Marquardt       (OIAC adaptive    (physics +
    leg θ_i)      (Cartesian foot        IK (joint        impedance →       rendering →
                   targets + stance/      angles)         joint torques)    sensors)
                   swing schedule)
```

| Stage | Module | Role |
|-------|--------|------|
| **CPG** | `python/cpg/kuramoto_cpg.py` | 4 coupled phase oscillators (one per leg) produce synchronised gait phases. |
| **Trajectory** | `python/cpg/trajectory_builder.py` | Maps phase → Cartesian foot target (ellipsoid path); `GaitScheduler` gives per-foot stance/swing. |
| **IK** | `python/inverse_kinematics/inverse_kin.py` | `LevenbergMarquardtIK` converts foot targets → 3-DOF joint angles per leg. |
| **PD / OIAC** | `python/pd/muscle_like_pd.py`, `python/pd/adaptive_imp.py` | Online Impedance Adaptation Control (Xiong & Fang, 2023) → joint torques. |
| **Orchestration** | `python/controllers/ik_controller.py` | Runs the CPG → IK → PD chain every control step. |
| **Simulation** | `python/sim/mujoco_sim.py` | Loads the Go2 model, applies torques, renders, and feeds sensors back. |
| **Learning** | `python/sim/gait_env.py`, `python/sim/reinforcement_learning.py` | Gymnasium env + PPO/CMA-ES to optimise gait parameters. |
| **Shared state** | `python/shared_module/` | `RobotInterface`, gait/joint/foot enums, constants, INI settings loader. |

**Supported gaits:** `WALK`, `TROT`, `AMBLE`, `CANTER`, `GALLOP` — each defined
by a parameter set in `data/.settings/`.

---

## Repository layout

```
bio_inspired_muscle/
├── python/                  # Active implementation (the project lives here)
│   ├── cpg/                 # Central Pattern Generators (Kuramoto, Matsuoka, trajectory builder)
│   │   └── brian_sim/       # Brian2 spiking-neuron CPG experiments
│   ├── controllers/         # IK controller — orchestrates the control pipeline
│   ├── inverse_kinematics/  # Levenberg-Marquardt leg IK solver
│   ├── pd/                  # Muscle-like PD + adaptive-impedance (OIAC) control
│   ├── fuzzy_logic/         # Fuzzy gait-switcher (velocity → gait selection)
│   ├── sim/                 # MuJoCo simulation, RL optimiser, demo/test scripts
│   │   └── go2/             # Unitree Go2 MJCF model, meshes, scenes, terrain generators
│   ├── shared_module/       # RobotInterface state hub, enums, constants, settings loader
│   ├── plotting/            # Time-series, heat-map and PD-vs-OIAC plotting utilities
│   ├── logger/              # Lightweight file + stdout logger
│   └── requirements.txt
├── cpp/                     # C++ scaffolding (CMake; not the active implementation)
├── data/                    # Gait configs (.settings/*.ini), RL result CSVs, output plots
├── scripts/                 # Terminal-UI launcher
├── Docs/                    # Papers, GET_STARTED guide, demo video, generated API docs
├── Doxyfile                 # Doxygen config (generates Docs/doxygen/html)
└── README.md
```

---

## Setup

Use **Python 3.12** and a virtual environment. MuJoCo ≥ 3.4.0 is required (it
ships its own viewer — modern MuJoCo has no `./simulate` binary).

```bash
git clone https://github.com/Toasting-Snowdragon-9999/bio_inspired_muscle.git
cd bio_inspired_muscle

# Create and activate a virtual environment
python3 -m venv python/.venv
source python/.venv/bin/activate          # Linux / macOS

# Install dependencies
pip install -r python/requirements.txt
```

Key dependencies: `mujoco`, `numpy`, `scipy`, `matplotlib`, `gymnasium`,
`stable-baselines3`, `cma`, `Brian2`, `opencv-python`.

---

## Running

All commands assume the virtual environment is active.

### Walk the robot (CPG-driven simulation)

```bash
cd python/sim
python mujoco_kuramoto_test.py
```

Opens the interactive MuJoCo viewer with the Go2 walking under CPG control.
Different terrains (flat, hills, rough) and gaits can be selected in the script.

### Manual joint control (debugging)

```bash
cd python/sim
python mujoco_manual_control.py        # keys select leg/joint and nudge the angle
```

### Visualise the CPG and foot trajectories

```bash
cd python/cpg
python test.py                         # matplotlib plots of oscillator traces + foot paths
```

### Optimise gait parameters with reinforcement learning

```bash
cd python/sim

# CMA-ES (evolution strategy) — 200 generations of the TROT gait
python reinforcement_learning.py --algo cma --gait TROT --n-generations 200

# PPO (deep RL)
python reinforcement_learning.py --algo ppo --gait TROT --total-timesteps 500

# Custom reward weights (Cost-of-Transport, velocity, body tilt …)
python reinforcement_learning.py --algo cma --w-cot 1.5 --w-vel 0.3 --w-tilt 3.0
```

Results are written to `data/rl_results_*.csv`. Plot the convergence with:

```bash
python data/visualize_convergence.py --group     # group runs by gait
```

---

## Gait configuration

Each gait is stored as an INI file in `data/.settings/` (e.g. `.WALK.ini`,
`.TROT.ini`, `.CANTER.ini`, `.GALLOP.ini`, `.AMBLE.ini`) and loaded by
`python/shared_module/settings_loader.py`. The parameters define the locomotion
pattern — for example:

```ini
[gait]
duty_factor    = 0.500000      # fraction of the cycle a foot is on the ground
frequency      = 2.200000      # stride frequency (Hz)
front_x_fore   = 0.123000      # forward swing distance (m)
front_z_top    = 0.103600      # swing height (m)
front_rotation = 0.045600      # hip-pitch rotation
...
```

`*_rl.ini` / `*_rl.pkl` files hold the best parameters found by the RL optimiser.

---

## API documentation (Doxygen)

Every class and function is documented with Doxygen-style `@brief` / `@param` /
`@return` tags. Generate browsable HTML docs with:

```bash
doxygen Doxyfile
# open Docs/doxygen/html/index.html
```

---

## C++ side

`cpp/` contains CMake scaffolding (`bio_inspired_locomotion`, `cpg`,
`pd_controller` targets). It is structural only — the active implementation is
in Python. See [`cpp/README.md`](cpp/README.md).

---

## Further reading

- **Project plan & conventions:** [`python/README.md`](python/README.md)
- **Getting-started guide & command reference:** [`Docs/GET_STARTED.md`](Docs/GET_STARTED.md), [`Docs/commands.md`](Docs/commands.md)
- **Papers** (in `Docs/`): adaptive impedance / OIAC (Xiong & Fang, 2023), synergy-based quadruped locomotion, CPG models (Matsuoka, eight-neuron), and learning control of emulated muscle.

---

*A robotics bachelor project. Target platform: Unitree Go2, simulated in MuJoCo.*
