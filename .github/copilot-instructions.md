# Copilot Instructions — bio_inspired_muscle

## Project Plan

Always read and follow the project plan in `python/README.md` before making changes.
The `python/README.md` describes the architecture, active files, research references, development roadmap, and coding conventions.
Treat `python/README.md` as the single source of truth for what to build and how.
The root `README.md` provides a short project overview.

## Scope

This project's primary codebase lives in the `python/` directory.
Focus assistance on Python code within `python/` and its sub-modules unless explicitly asked otherwise.

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
