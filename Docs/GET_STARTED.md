# Getting Started — Bio-Inspired Muscle Control

Welcome to the **bio_inspired_muscle** project! This is a bio-inspired locomotion controller for a **Unitree Go2 quadruped robot**, simulated in **MuJoCo**. The control pipeline mimics biological motor patterns: a Central Pattern Generator (CPG) produces rhythmic signals, which are converted to foot trajectories, solved with inverse kinematics, and driven by adaptive muscle-like PD controllers.

This guide will walk you through the project structure, explain each component, show how they connect, and get you running the simulation.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Project Structure Overview](#2-project-structure-overview)
3. [The Control Pipeline (Big Picture)](#3-the-control-pipeline-big-picture)
4. [Module Deep Dives](#4-module-deep-dives)
   - [Shared Module — The Backbone](#41-shared-module--the-backbone)
   - [CPG — Locomotion Rhythm](#42-cpg--locomotion-rhythm)
   - [Trajectory Builder — Phase to Foot Paths](#43-trajectory-builder--phase-to-foot-paths)
   - [Inverse Kinematics — Cartesian to Joint Space](#44-inverse-kinematics--cartesian-to-joint-space)
   - [PD Control — Adaptive Muscle-Like Torques](#45-pd-control--adaptive-muscle-like-torques)
   - [IK Controller — The Orchestrator](#46-ik-controller--the-orchestrator)
   - [MuJoCo Simulation — Physics & Rendering](#47-mujoco-simulation--physics--rendering)
5. [Running the Simulation](#5-running-the-simulation)
6. [Debugging & Visualization Tools](#6-debugging--visualization-tools)
7. [Research Papers & References](#7-research-papers--references)
8. [Key Parameters & Tuning Guide](#8-key-parameters--tuning-guide)
9. [Coding Conventions & House Rules](#9-coding-conventions--house-rules)
10. [Common Pitfalls](#10-common-pitfalls)

---

## 1. Prerequisites & Environment Setup

### System Requirements

- **Python 3.12.3**
- A display server (X11 or Wayland) — MuJoCo renders via GLFW
- OpenGL support (for the simulation viewer)

### Setting Up the Virtual Environment

The project uses a virtual environment located at `python/.bach_env/`. To set it up:

```bash
cd python/
python3 -m venv .bach_env
source .bach_env/bin/activate
pip install -r requirements.txt
```

### Dependencies

| Package      | Version  | Purpose                                    |
|-------------|----------|--------------------------------------------|
| `mujoco`    | ≥ 3.4.0  | Physics simulation engine                  |
| `numpy`     | ≥ 2.4.2  | Vectorised math everywhere                 |
| `scipy`     | ≥ 1.17.0 | Numerical utilities                        |
| `matplotlib`| ≥ 3.10.8 | Real-time plotting and debugging graphs    |
| `Brian2`    | ≥ 2.10.1 | Neural oscillator simulations (reference)  |

---

## 2. Project Structure Overview

```
bio_inspired_muscle/
├── Docs/                         # Research papers (PDF)
│   ├── cpg_related/              #   CPG theory (Matsuoka, Kuramoto)
│   ├── pd_related/               #   Adaptive impedance control papers
│   └── fuzzy_logic_related/      #   Fuzzy logic (planned, future work)
├── python/                       # ← All active code lives here
│   ├── cpg/                      #   Central Pattern Generators
│   │   ├── kuramoto_cpg.py       #     Kuramoto oscillator (PRIMARY CPG)
│   │   ├── matsouka_cpg.py       #     Matsuoka oscillator (incomplete stub)
│   │   ├── trajectory_builder.py #     Phase → Cartesian foot trajectories
│   │   ├── test.py               #     CPG & trajectory test/visualization
│   │   └── brian_sim/            #     Brian2 neural simulations (reference)
│   ├── pd/                       #   PD / impedance controllers
│   │   ├── muscle_like_pd.py     #     Per-leg PD wrapper
│   │   ├── adaptive_imp.py       #     Online impedance adaptation (IOAC)
│   │   └── balancing_control.py  #     Abductor control (stub, not implemented)
│   ├── controllers/              #   High-level controllers
│   │   └── ik_controller.py      #     MAIN pipeline: CPG→Traj→IK→PD→Torque
│   ├── inverse_kinematics/       #   Leg IK solver
│   │   └── inverse_kin.py        #     Analytical FK + Jacobian + LM-IK
│   ├── sim/                      #   Simulation & test scenes
│   │   ├── mujoco_sim.py         #     Core simulation engine & renderer
│   │   ├── mujoco_kuramoto_test.py #   ★ MAIN ENTRY POINT ★
│   │   ├── mujoco_manual_control.py #  Manual joint debug tool
│   │   ├── go2/                  #     Unitree Go2 MuJoCo model
│   │   │   ├── scene.xml         #       Flat ground scene
│   │   │   ├── scene_terrain.xml #       Rough terrain scene
│   │   │   ├── scene_stairs.xml  #       Stairs scene
│   │   │   └── go2.xml           #       Robot model definition
│   │   └── ...                   #     Other test scripts
│   ├── shared_module/            #   Enums, constants, robot state
│   │   ├── robot_state.py        #     RobotInterface + enums (backbone)
│   │   └── global_constants.py   #     Joint ranges, actuator maps, limits
│   ├── plotting/                 #   Visualization (not actively used)
│   └── requirements.txt
├── scripts/                      # Utility scripts (currently empty)
└── README.md                     # Project overview
```

> **Files prefixed with `(old)`** (e.g., `(old)_muscle_like_pd.py`) are legacy — do not modify or import them.

---

## 3. The Control Pipeline (Big Picture)

Every MuJoCo timestep (~0.002s), the controller runs the following chain:

```
┌────────────────────────────────────────────────────────────────────┐
│                    IKController.run()  (called every timestep)     │
│                                                                    │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ Kuramoto CPG │───▶│ Trajectory       │───▶│ Inverse          │  │
│  │              │    │ Builder          │    │ Kinematics       │  │
│  │ 4 coupled    │    │                  │    │                  │  │
│  │ oscillators  │    │ Phase → Cartesian│    │ Cartesian →      │  │
│  │              │    │ foot positions   │    │ joint angles     │  │
│  │ Output:      │    │ & velocities     │    │ & velocities     │  │
│  │ θ[4],  dθ[4] │    │ (hip frame)      │    │ (per leg)        │  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                       │            │
│                                                       ▼            │
│                                              ┌──────────────────┐  │
│                                              │ Muscle-Like PD   │  │
│                                              │ Controller       │  │
│                                              │                  │  │
│                                              │ Joint errors →   │  │
│                                              │ adaptive torques │  │
│                                              │ (clipped ±23.7Nm)│  │
│                                              └────────┬─────────┘  │
│                                                       │            │
│                                                       ▼            │
│                                              ┌──────────────────┐  │
│                                              │ RobotInterface   │  │
│                                              │ .target_torques  │  │
│                                              └────────┬─────────┘  │
└───────────────────────────────────────────────────────┼────────────┘
                                                        │
                                                        ▼
                                              ┌───────────────────┐
                                              │ MuJoCo Physics    │
                                              │ Engine            │
                                              │ + bias comp.      │
                                              │ + actuator limits │
                                              └───────────────────┘ 
```

**In plain English:**
1. The **CPG** generates rhythmic phase signals (like a biological spinal cord).
2. The **Trajectory Builder** turns those phases into desired foot positions in 3D space.
3. The **IK solver** converts foot positions to joint angles the robot can use.
4. The **PD controller** computes motor torques that drive the joints toward those angles.
5. **MuJoCo** applies the torques, simulates physics, and feeds back joint state for the next cycle.

---

## 4. Module Deep Dives

### 4.1 Shared Module — The Backbone

> **Files:** `shared_module/robot_state.py`, `shared_module/global_constants.py`

These two files are imported by nearly everything else. They define the common language of the project.

#### `robot_state.py` — Central State & Communication Hub

This is the **single most important file** to understand. The `RobotInterface` class is the shared data bus between the simulation and the controller.

**Key Enums:**
| Enum   | Values | Purpose |
|--------|--------|---------|
| `Gait` | Walk, Trot, Bound, Pace, Gallop | Defines phase offsets between legs |
| `Mode` | Moving, Transition | Current locomotion mode |
| `Joint`| 12 values (e.g. `FR_HIP`, `FL_THIGH`, `RR_CALF`) | Identifies every actuated joint |
| `Foot` | FL, FR, RL, RR | Identifies legs (also carries MuJoCo body names) |

**`State` dataclass:**
```python
@dataclass
class State:
    mode: Mode        # Moving or Transition
    gait: Gait        # Which gait pattern (Trot, Walk, etc.)
    frequency: float  # CPG base frequency in Hz
```

**`RobotInterface` class — what it holds:**
```python
robot_interface.joint_positions    # dict[Joint, float] — current angles (from MuJoCo sensors)
robot_interface.joint_velocities   # dict[Joint, float] — current velocities
robot_interface.body_position      # [x, y, z] — torso position in world
robot_interface.body_orientation   # [qw, qx, qy, qz] — torso orientation
robot_interface.foot_positions     # dict[Foot, [x,y,z]] — foot positions in world
robot_interface.target_torques     # dict[Joint, float] — SET by controller, READ by sim
robot_interface.dt                 # float — simulation timestep
robot_interface.state              # RobotState — current/previous/next state
```

**Data flow pattern:** The simulation writes sensor data → `RobotInterface` → the controller reads it, computes, and writes torques back → the simulation reads torques and applies them.

#### `global_constants.py` — Configuration & Mappings

This file is the lookup table for everything hardware-related:

- **Joint ranges** (radians): e.g. knee can go from -2.72 to -0.84
- **`ACTUATOR_DICT`**: Maps `Joint` enum → MuJoCo actuator index (0–11)
- **`SENSOR_POS_DICT` / `SENSOR_VEL_DICT`**: Maps `Joint` → sensor indices
- **`NEURON_TO_FOOT_DICT`**: Maps CPG oscillator index (0–3) → `Foot`
- **`FOOT_TO_JOINT_DICT`**: Maps `Foot` → list of 3 `Joint` enums (hip, thigh, calf)
- **`GAIT_ENERGY`**: Recommended frequency per gait (Walk=0.5Hz, Trot=0.7Hz, etc.)
- **`ACTUATOR_TORQUE_LIMIT`**: 23.7 Nm (motor max)

---

### 4.2 CPG — Locomotion Rhythm

> **File:** `cpg/kuramoto_cpg.py` (primary), `cpg/matsouka_cpg.py` (stub, not used)

A **Central Pattern Generator** is a neural circuit that produces rhythmic output without needing sensory feedback — think of it as the "heartbeat" of locomotion.

#### How It Works

The project uses a **Kuramoto oscillator** model. There are 4 oscillators (one per leg), each described by a single phase variable θᵢ:

```
dθᵢ/dt = ωᵢ + Σⱼ wᵢⱼ · sin(θⱼ − θᵢ − φᵢⱼ)
```

Where:
- **ωᵢ** = base frequency (e.g. 1.5 Hz × 2π for trot) — read from `RobotInterface.state.frequency`
- **wᵢⱼ** = coupling strength between oscillators (default: 1.5π) — how strongly legs influence each other
- **φᵢⱼ** = desired phase offset between legs — **this is what defines the gait**

**Gait = phase offsets between legs:**

| Gait   | Pattern | Phase offsets (FL, FR, RL, RR) |
|--------|---------|-------------------------------|
| Trot   | Diagonal pairs in sync | (0, π, π, 0) |
| Walk   | Sequential | (0, π, π/2, 3π/2) |
| Bound  | Front pair + back pair | (0, 0, π, π) |
| Pace   | Same-side pairs | (0, π, 0, π) |

The oscillators are integrated using **RK4** (4th-order Runge-Kutta) each timestep.

**Outputs:**
- `get_phase_outputs()` → 4 phase values (θ₀, θ₁, θ₂, θ₃) — used for foot position
- `get_phase_velocities()` → 4 phase derivatives (dθ/dt) — used for foot velocity

---

### 4.3 Trajectory Builder — Phase to Foot Paths

> **File:** `cpg/trajectory_builder.py`

The trajectory builder converts abstract CPG phase signals into concrete 3D foot positions that the robot's legs should follow.

#### The Math

For each foot, given phase θ and phase velocity dθ/dt:

**Position (in hip-local frame):**
```
x = -width · cos(θ)                    # forward/backward stride
z = amplitude · sin(θ)                  # up/down foot lift
y = 0                                   # lateral (locked to zero)
```

The z-amplitude **blends** between two values depending on whether the foot is in swing or stance:
- **Swing** (foot in air): amplitude = `step_height` (e.g. 0.15m)
- **Stance** (foot on ground): amplitude = `stance_depth` (0.02m, slight ground compliance)

The blending uses a smooth tanh sigmoid based on sin(θ):
```
blend = 0.5 · (1 + tanh(blend_sharpness · sin(θ)))
amplitude = blend · step_height + (1 - blend) · stance_depth
```

**Velocity (analytical derivatives):**
```
dx/dt = width · sin(θ) · dθ/dt
dz/dt = amplitude · cos(θ) · dθ/dt
```

**Key Parameters:**
| Parameter | Default | Effect |
|-----------|---------|--------|
| `width` (stride_length) | 0.06–0.1 m | How far forward/back each foot swings |
| `step_height` | 0.04–0.15 m | How high the foot lifts in swing |
| `stance_depth` | 0.02 m | Slight foot press into ground during stance |
| `blend_sharpness` | 10.0 | How abruptly swing↔stance transitions (higher = sharper) |

---

### 4.4 Inverse Kinematics — Cartesian to Joint Space

> **File:** `inverse_kinematics/inverse_kin.py`

The IK module converts "put the foot here in 3D space" into "set these 3 joint angles." Each Go2 leg has 3 degrees of freedom: **abduction** (hip lateral), **hip pitch**, and **knee pitch**.

#### Robot Geometry

```
         hip (abduction)
          │
          ├── thigh (hip pitch)    L_THIGH = 0.213 m
          │
          └── calf (knee pitch)    L_CALF  = 0.213 m
              │
              foot
```

#### Key Components

1. **`forward_kinematics(q, d_y)`** — Given joint angles, compute foot position:
   ```
   x = -L_thigh·sin(q_hip) - L_calf·sin(q_hip + q_knee)
   z = -L_thigh·cos(q_hip) - L_calf·cos(q_hip + q_knee)
   y = d_y  (lateral offset, depends on which leg)
   ```

2. **`leg_jacobian(q, d_y)`** — 3×3 matrix mapping joint velocities to foot velocities:
   ```
   J = ∂(foot_position) / ∂(joint_angles)
   ```
   Used to compute: `dq = J⁻¹ · dx` (what joint velocities achieve desired foot velocity)

3. **`LevenbergMarquardtIK`** — Iterative IK solver (damped least-squares):
   ```
   for each iteration:
       error = goal - FK(q)
       J = Jacobian(q)
       Δq = (JᵀJ + λI)⁻¹ · Jᵀ · error    # damped least-squares step
       q += step_size · Δq
       if ||error|| < tolerance: break
   ```
   - Clips results to joint limits
   - Warm-starts from previous solution for speed (via `init_q` parameter)

---

### 4.5 PD Control — Adaptive Muscle-Like Torques

> **Files:** `pd/muscle_like_pd.py`, `pd/adaptive_imp.py`

This is where joint angle errors become motor torques. The controller mimics biological muscle properties with adaptive stiffness and damping.

#### `muscle_like_pd.py` — Per-Leg PD Wrapper

For each of the 4 legs, every timestep:

```python
e  = q_desired - q_actual       # position error (3 joints)
de = dq_desired - dq_actual     # velocity error (3 joints)

K, B = adaptive_impedance(...)  # stiffness & damping matrices (3×3)

torque = K @ e + B @ de         # PD law with adaptive gains
torque = clip(torque, ±23.7)    # actuator limit
```

#### `adaptive_imp.py` — Online Impedance Adaptation (IOAC)

Based on **Xiong & Fang (2023)**. Two modes:

**Fixed mode** (`use_ioac=False`): Simple diagonal PD gains
```
K = diag(90, 90, 90)   # stiffness
B = diag(15, 15, 15)   # damping
```

**Adaptive mode** (`use_ioac=True`): Gains adapt based on tracking errors
```
tracking_error = k · velocity_error + position_error    # k=0.05
adaptation_factor = a / (1 + b · ||tracking_error||²)   # a=0.2, b=5.0

K = (tracking_error ⊗ position_error) / adaptation_factor
B = (tracking_error ⊗ velocity_error) / adaptation_factor
```

This means: when tracking error is large, gains increase automatically (stiffer response). When the robot tracks well, gains relax (more compliant). This mimics how biological muscles modulate their impedance.

---

### 4.6 IK Controller — The Orchestrator

> **File:** `controllers/ik_controller.py`

This is the **main control class** that wires everything together. Its `run()` method is called every simulation timestep.

#### `IKController.run()` step-by-step:

```python
def run(self):
    # 1. Advance CPG by one timestep
    self.cpg.run()
    phases     = self.cpg.get_phase_outputs()      # [θ_FL, θ_FR, θ_RL, θ_RR]
    velocities = self.cpg.get_phase_velocities()    # [dθ_FL, dθ_FR, dθ_RL, dθ_RR]

    # 2. Build foot trajectories from phase signals
    foot_pos, foot_vel = self.traj_builder.build_trajectory(phases, velocities)
    # foot_pos: dict[Foot, Coordinate(x, y, z)]  — target foot positions in hip frame
    # foot_vel: dict[Foot, Coordinate(x, y, z)]  — target foot velocities

    # 3. For each leg, solve IK and compute joint velocities
    for foot in [Foot.FL, Foot.FR, Foot.RL, Foot.RR]:
        target = convert_frame(foot_pos[foot], foot)    # frame adjustment
        q = self.solvers[foot].calculate(target, init_q=self.prev_q[foot])  # IK solve
        
        J = leg_jacobian(q_array, d_y)                  # 3×3 Jacobian
        dq = np.linalg.pinv(J) @ v_cartesian            # velocity IK
        dq = np.clip(dq, -MAX_JOINT_VEL, MAX_JOINT_VEL) # clamp velocities
        
        # Zero out abduction (not controlled in sagittal plane)
        q[hip_joint] = 0.0
        dq[0] = 0.0

    # 4. Compute torques via PD controller
    torques = self.pd.control(q_targets, dq_targets)    # dict[Foot, ndarray(3,)]

    # 5. Send to robot
    robot_interface.target_positions = q_targets    # for monitoring
    robot_interface.target_torques = torques         # for actuation
```

**Key parameters:**
- `stride_length`: How far feet swing forward/back (default: 0.06m)
- `step_height`: Dict of per-leg swing heights (front legs often higher than rear)
- `MAX_JOINT_VEL`: 5.0 rad/s velocity clamp
- `use_adaptive_pd`: Toggle adaptive vs fixed PD gains

---

### 4.7 MuJoCo Simulation — Physics & Rendering

> **File:** `sim/mujoco_sim.py`

The simulation engine handles everything MuJoCo: loading the robot model, running physics, rendering graphics, and bridging data between MuJoCo and the control pipeline.

#### `MujocoSim` — What it does

1. **Loads the Go2 model** from `sim/go2/scene.xml`
2. **Syncs sensor data** → `RobotInterface` every timestep (`_sync_robot_interface()`)
   - Reads joint positions/velocities from MuJoCo sensors
   - Reads body/foot/hip positions in world frame
3. **Applies controller torques** to MuJoCo actuators (`_apply_controller_targets_torque()`)
   - Adds **bias compensation** (gravity + Coriolis forces)
   - Clips to actuator limits
4. **Renders** the 3D scene via GLFW/OpenGL
5. **Tracks energy** for Cost of Transport calculation

#### The Simulation Loop

```python
sim.sim(controller, sim_length=-1, slow_factor=1.0)
```

Internally, MuJoCo calls a **control callback** at each physics step:
```
_control_callback():
    1. _sync_robot_interface()         # MuJoCo sensors → RobotInterface
    2. controller.run()                # IKController does its thing
    3. _apply_controller_targets_torque()  # RobotInterface → MuJoCo actuators
```

The rendering loop targets 60 FPS. `slow_factor` can slow things down for observation.

#### MuJoCo Scenes

| Scene | File | Description |
|-------|------|-------------|
| Flat ground | `go2/scene.xml` | Standard test surface |
| Terrain | `go2/scene_terrain.xml` | Bumpy/uneven ground |
| Stairs | `go2/scene_stairs.xml` | Step climbing |

---

## 5. Running the Simulation

### Quick Start

```bash
cd python/
source .bach_env/bin/activate

# Run the main integration test (trot gait, full pipeline)
cd sim/
python mujoco_kuramoto_test.py
```

This launches a 3D MuJoCo window showing the Go2 robot trotting. Controls:
- **Mouse drag**: Rotate camera
- **Scroll**: Zoom
- **Backspace**: Reset simulation

### What `mujoco_kuramoto_test.py` Does

```python
# 1. Define initial state: trot gait at 1.5 Hz
state = State(mode=Mode.MOVING, gait=Gait.TROT, frequency=1.5)
robot_interface = RobotInterface(state)

# 2. Create simulation with Go2 model
sim = MujocoSim('go2/scene.xml', robot_interface, window_scale=2.0)

# 3. Create controller with gait parameters
step_height = {Foot.FL: 0.15, Foot.FR: 0.15, Foot.RL: 0.1, Foot.RR: 0.1}
controller = IKController(robot_interface, stride_length=0.1,
                          step_height=step_height, use_adaptive_pd=True)

# 4. Run simulation indefinitely (sim_length=-1)
sim.sim(controller=controller, sim_length=-1, slow_factor=1.0)

# 5. Print Cost of Transport when window is closed
print("Cost of Transport:", sim.compute_CoT())
```

### Other Test Scripts

| Script | Purpose | How to run |
|--------|---------|------------|
| `mujoco_manual_control.py` | Manually move individual joints with keyboard | `python mujoco_manual_control.py` |
| `mujoco_test.py` | Basic MuJoCo loading test | `python mujoco_test.py` |
| `cpg/test.py` | Visualise CPG outputs and foot trajectories (matplotlib) | `cd ../cpg && python test.py` |

---

## 6. Debugging & Visualization Tools

The simulation has several built-in debug modes you can enable in the test script:

```python
sim = MujocoSim(xml_path, robot_interface, window_scale=2.0)

sim.enable_air_mode(0.5)      # Lift robot 0.5m — test leg swing without ground contact
sim.enable_graph()             # Show real-time oscillator output plots (hip/knee curves)
sim.enable_joint_graph()       # Show all 12 joint position traces
sim.enable_joint_sliders()     # Interactive joint angle sliders
sim.set_bias_compensation(False) # Disable gravity compensation (to test raw torques)
sim.set_camera_follow(True)    # Camera tracks the robot
```

### CPG Standalone Visualization

```bash
cd python/cpg/
python test.py
```

This generates matplotlib plots of:
- Oscillator outputs over time
- Foot trajectories in the X-Z plane (sagittal view)
- 3D foot paths with velocity vectors
- Phase-space plots

---

## 7. Research Papers & References

The control approach is grounded in published research. Papers are in `Docs/`:

| Paper / Topic | Location | Used in |
|---------------|----------|---------|
| **Kuramoto oscillators** (coupled phase model) | `Docs/cpg_related/` | `cpg/kuramoto_cpg.py` |
| **Matsuoka (1985)** — half-centre oscillator | `Docs/cpg_related/` | `cpg/matsouka_cpg.py` (stub) |
| **Hexel (2015)** — eight-neuron CPG network | `Docs/cpg_related/eight_neuron/` | Reference architecture |
| **Xiong & Fang (2023)** — online impedance adaptation | `Docs/pd_related/ioac_paper.pdf` | `pd/adaptive_imp.py` |
| **Muscle-like learning control** | `Docs/pd_related/` | `pd/muscle_like_pd.py` |
| **Fuzzy logic control** (future) | `Docs/fuzzy_logic_related/` | Not yet implemented |

When working on a module, **read the corresponding paper first** — the code directly implements equations from these papers.

---

## 8. Key Parameters & Tuning Guide

### Gait Parameters

| Parameter | Location | Default | What it does |
|-----------|----------|---------|--------------|
| `gait` | `State()` constructor | `Gait.TROT` | Phase coupling pattern |
| `frequency` | `State()` constructor | 1.5 Hz | CPG oscillation speed |
| `stride_length` | `IKController()` | 0.06–0.1 m | Forward/back foot excursion |
| `step_height` | `IKController()` | 0.04–0.15 m | Foot lift height (per leg) |
| `coupling_strength` | `kuramoto_cpg.py` | 1.5π | How tightly legs synchronize |

### PD / Impedance Parameters

| Parameter | Location | Default | What it does |
|-----------|----------|---------|--------------|
| `kp` (fixed mode) | `adaptive_imp.py` | 90 | Proportional gain (stiffness) |
| `kd` (fixed mode) | `adaptive_imp.py` | 15 | Derivative gain (damping) |
| `a` (adaptive) | `adaptive_imp.py` | 0.2 | Adaptation rate numerator |
| `b` (adaptive) | `adaptive_imp.py` | 5.0 | Adaptation sensitivity denominator |
| `k` (adaptive) | `adaptive_imp.py` | 0.05 | Velocity error weight in tracking error |

### Physical Limits

| Parameter | Value | Source |
|-----------|-------|--------|
| `ACTUATOR_TORQUE_LIMIT` | 23.7 Nm | `global_constants.py` |
| `MAX_JOINT_VEL` | 5.0 rad/s | `ik_controller.py` |
| Knee range | -2.72 to -0.84 rad | `global_constants.py` |
| Hip range (front) | -1.57 to 3.49 rad | `global_constants.py` |

### Tuning Tips

- **Robot falls over?** Lower `frequency`, reduce `stride_length`, or switch to fixed PD (`use_adaptive_pd=False`).
- **Legs look sluggish?** Increase `MAX_JOINT_VEL` or reduce CPG `frequency` (velocity scales with dθ/dt).
- **Jerky ground contacts?** Reduce adaptive `a` parameter or increase `b` to soften impedance changes.
- **Use `sim.enable_air_mode(0.5)`** to test leg swing without ground contact — great for tuning trajectory shape.

---

## 9. Coding Conventions & House Rules

| Rule | Details |
|------|---------|
| **Type hints** | Always on function signatures |
| **Dataclasses** | For structured parameter groups |
| **NumPy arrays** | For vectorised math — no Python loops over joints |
| **Enums** | Use `Joint`, `Foot`, `Gait`, `State` from `robot_state.py` — never raw strings/ints |
| **Constants** | All go in `global_constants.py` |
| **Comments** | **Never remove or modify existing comments** — they document design decisions and paper references |
| **`(old)` files** | Legacy — do not modify or import |
| **Cross-module imports** | Use the `sys.path.insert` pattern already in the codebase |
| **Virtual env** | `python/.bach_env/` — never commit venv files |
| **File deletion** | Not allowed without explicit permission — flag obsolete files in README instead |

---

## 10. Common Pitfalls

1. **Import paths**: Modules use `sys.path.insert(0, ...)` for cross-directory imports. If you get `ModuleNotFoundError`, check that you're running from the correct directory (usually `python/sim/` for simulations or `python/cpg/` for CPG tests).

2. **`matsouka_cpg.py` is NOT the active CPG**: Despite the name suggesting it's the Matsuoka oscillator, the active CPG is `kuramoto_cpg.py`. The Matsuoka file is an incomplete stub.

3. **Abduction joints are zeroed**: The IK controller only controls the sagittal plane (hip pitch + knee). Hip abduction (lateral) is explicitly set to 0. Don't expect lateral foot movement.

4. **Velocity magnitude scales with frequency**: At 1.5 Hz, dθ/dt ≈ 9.4 rad/s, producing Cartesian velocities up to ~0.9 m/s. This can saturate the `MAX_JOINT_VEL` clamp.

5. **Adaptive impedance can be aggressive**: The IOAC algorithm computes K and B as outer products. Large velocity errors at ground contact can spike torques. If the robot explodes, try fixed PD first.

6. **Display required**: MuJoCo renders via GLFW/OpenGL. If running headless (SSH, CI), you'll need a virtual display (`xvfb-run python ...`).

7. **Brian2 is reference only**: The `cpg/brian_sim/` directory contains neural oscillator experiments using the Brian2 spiking neural network library. These are research explorations, not part of the control pipeline.

---

## Quick Reference: "Where Do I Find...?"

| I want to... | Go to... |
|---------------|----------|
| Run the robot simulation | `python/sim/mujoco_kuramoto_test.py` |
| Change the gait or frequency | `State()` constructor in the test script |
| Tune stride/step height | `IKController()` constructor in the test script |
| Change PD gains | `python/pd/adaptive_imp.py` |
| Modify foot trajectory shape | `python/cpg/trajectory_builder.py` |
| Change CPG coupling/oscillators | `python/cpg/kuramoto_cpg.py` |
| Add a new joint mapping or constant | `python/shared_module/global_constants.py` |
| Understand the robot state interface | `python/shared_module/robot_state.py` |
| Check joint limits or actuator indices | `python/shared_module/global_constants.py` |
| See the robot model (URDF/MJCF) | `python/sim/go2/go2.xml` |
| Read the detailed project plan | `python/README.md` |
