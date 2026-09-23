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

A recording of the result — the Go2 locomoting in MuJoCo — is in the repository:

▶️ [`Docs/Quadruped_locomotion.mp4`](Docs/Quadruped_locomotion.mp4) (5 MB)

> GitHub does not play `.mp4` files inline from the repository tree: click the
> link above, then **View raw** to download and play it locally. To embed it as
> a player in this README instead, drag the file into a GitHub issue or PR
> comment box and paste the `user-attachments` URL GitHub returns.

---

## Architecture

The controller is a feed-forward pipeline driven by a shared state object
(`RobotInterface`). Each control step flows top-to-bottom; sensor readings flow
back into the shared state:

```
                ┌──────────────────────────────────────────────┐
                │  RobotInterface  (central shared state hub)   │
                │  gait, frequency, joint pos/vel, contacts …   │
                └───────────────┬───────────────────▲──────────┘
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
│   ├── cpg/                 # Central Pattern Generators (Kuramoto, trajectory builder)
│   │   └── brian_sim/       # Brian2 spiking-neuron CPG experiments (standalone)
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
├── scripts/                 # Terminal-UI launcher (POSIX only)
├── Docs/                    # Papers, GET_STARTED guide, demo video, generated API docs
├── Doxyfile                 # Doxygen config (generates Docs/doxygen/html)
└── README.md
```

---

## Setup

Requires **Python 3.12 or newer** (developed on 3.12, verified on 3.14) and
MuJoCo ≥ 3.4.0. Modern MuJoCo ships its own viewer — there is no `./simulate`
binary to build.

```bash
git clone https://github.com/Toasting-Snowdragon-9999/bio_inspired_muscle.git
cd bio_inspired_muscle
```

Create and activate a virtual environment:

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Then install the dependencies (same on all platforms):

```bash
pip install -r python/requirements.txt
```

Key dependencies: `mujoco`, `numpy`, `scipy`, `matplotlib`, `gymnasium`,
`stable-baselines3`, `cma`, `opencv-python`. `Brian2` is only needed for the
spiking-neuron experiments in `python/cpg/brian_sim/`.

### Verify the installation

This runs one four-second TROT gait headlessly (no window) and prints the
Cost of Transport. It should finish in well under a minute:

```bash
cd python/sim && python -c "from mujoco_kuramoto_test import *; cfg,f,d,p=load_settings_from_file(Gait.TROT); ri=RobotInterface(starting_state=State(mode=Mode.MOVING,gait=Gait.TROT,frequency=f),trajectory_method=TrajectoryMethod.ELLIPSOID,duty_factor=d); s=MujocoSim(get_sim_xml(Terrain.flat),robot_interface=ri,window_scale=2.0); set_spawn(s,Terrain.flat); c=IKController(robot_interface=ri,stride_length=None,step_height=None,params=p,use_adaptive_pd=True,ellipsoid_config=cfg); print('CoT:', s.headless_sim(controller=c,sim_length=4.0,warmup=1.0))"
```

Expected output: `CoT: 0.279…` (small variations are normal).

---

## Running

All commands assume the virtual environment is active. Run each script **from
the directory shown** — the scripts resolve their imports relative to their own
location.

### Walk the robot (CPG-driven simulation)

```bash
cd python/sim
python mujoco_kuramoto_test.py
```

Opens the interactive MuJoCo viewer with the Go2 walking under CPG control.
The gait, terrain and controller are chosen near the top of `elip_traj_test()`
in that file:

```python
gait    = Gait.TROT       # WALK | TROT | AMBLE | CANTER | GALLOP
terrain = Terrain.flat    # flat | tiny_hills | rough | very_rough | large_hills | tricky
...
controller = IKController(..., use_adaptive_pd=False)   # True = OIAC, False = fixed-gain PD
```

### Manual joint control (debugging)

```bash
cd python/sim
python mujoco_manual_control.py        # keys select leg/joint and nudge the angle
```

### Visualise the CPG and foot trajectories

```bash
cd python/cpg
python test.py                         # matplotlib plots of foot paths for the TROT gait
```

Other demos in `test.py` (phase space, 3-D trajectories, footfall diagrams) are
importable functions; they are not run by default because each opens a blocking
matplotlib window.

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

Results are written to `data/rl_results_*.csv`. Plot the convergence from the
repository root:

```bash
cd data
python visualize_convergence.py --group        # group runs by gait
```

---

## Gait configuration

Each gait is stored as an INI file in `data/.settings/` (`.WALK.ini`,
`.TROT.ini`, `.CANTER.ini`, `.GALLOP.ini`, `.AMBLE.ini`) and loaded by
`python/shared_module/settings_loader.py`. A file has two sections — `[gait]`
defines the foot-trajectory shape and timing, `[oiac]` the adaptive-impedance
gains:

```ini
[gait]
duty_factor    = 0.500000      # fraction of the cycle a foot is on the ground
frequency      = 2.200000      # stride frequency (Hz)
front_x_fore   = 0.123000      # forward swing distance (m)
front_z_top    = 0.103600      # swing height (m)
front_rotation = 0.045600      # hip-pitch rotation
...

[oiac]
a = 0.1                        # learning rate of the impedance adaptation
b = 20.0                       # sensitivity to velocity error
k = 0.07                       # baseline stiffness
```

If a file has no `[oiac]` section the controller falls back to its built-in
defaults. `*_rl.ini` / `*_rl.pkl` files hold the best parameters found by the
RL optimiser.

---

## API documentation (Doxygen)

Every class and function is documented with Doxygen-style `@brief` / `@param` /
`@return` tags. Generate browsable HTML docs with:

```bash
doxygen Doxyfile
# then open Docs/doxygen/html/index.html
```

---

## Known limitations

- **Trajectory shapes:** `TrajectoryMethod` declares `EGG`, `OVAL` and `BEZIER`
  alongside `ELLIPSOID`, but only `ELLIPSOID` is implemented. Selecting another
  raises `NotImplementedError`. `ELLIPSOID` is the default.
- **Fuzzy gait switching** (`python/fuzzy_logic/`) is implemented and testable
  standalone, but is not yet wired into the main simulation loop.
- **Terminal-UI launcher** (`scripts/interface_app.py`) depends on
  `simple_term_menu`, which is POSIX-only and does not run on Windows.
- **C++ side** (`cpp/`) is CMake scaffolding only — the active implementation is
  Python. See [`cpp/README.md`](cpp/README.md).
- **CANTER** runs but is the least-tuned gait; it makes noticeably less forward
  progress per stride than `TROT`.

---

## Further reading

- **Project plan & conventions:** [`python/README.md`](python/README.md)
- **Getting-started guide & command reference:** [`Docs/GET_STARTED.md`](Docs/GET_STARTED.md), [`Docs/commands.md`](Docs/commands.md)
- **Papers** (in `Docs/`): adaptive impedance / OIAC (Xiong & Fang, 2023), synergy-based quadruped locomotion, CPG models (Matsuoka, eight-neuron), and learning control of emulated muscle.

---

*A robotics bachelor project. Target platform: Unitree Go2, simulated in MuJoCo.*
