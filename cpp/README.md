# C++ scaffolding

This directory holds the **C++ scaffolding** for the project. The active
implementation lives in [`../python`](../python); the C++ side is structural
groundwork for a possible native port and is **not required** to run the
simulation or the controllers.

## Layout

```
cpp/
├── CMakeLists.txt      # CMake build (C++23)
├── include/            # Public headers
├── src/
│   ├── project_main.cpp  # entry point → bio_inspired_locomotion target
│   ├── cpg_main.cpp      # entry point → cpg target (CPG prototype)
│   └── pd_main.cpp       # entry point → pd_controller target (PD prototype)
└── build/              # Out-of-source build output (git-ignored)
```

## Build

```bash
cd cpp
cmake -S . -B build
cmake --build build
```

This produces three executables: `bio_inspired_locomotion`, `cpg`, and
`pd_controller`.
