# Repository Guidelines

## Project Structure & Module Organization

This directory is the GOAT robot's ROS 2 workspace.

- `src/goat_control/`: Python nodes, controller/motor/IMU utilities, YAML configuration, and launch files. Implementation lives in `goat_control/nodes/` and `goat_control/utils/`.
- `src/goat_control_cpp/`: C++ hardware implementation under migration; headers in `include/goat_control_cpp/`, implementations in `src/`, including vendored serial code.
- `src/goat_api/`: custom ROS message definitions and their CMake build.
- `src/goat_description/`: robot URDF, mesh assets, RViz configuration, and visualization launch files.
- `src/goat_sysid/`: system identification tools, experiment configuration, and recorded logs.
- `setup_can_*.sh`: host CAN configuration scripts. Consult `cpp_migration.md` for migration context.

## Build, Test, and Development Commands

Run from this directory with your installed ROS 2 environment sourced:

```bash
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select goat_control goat_control_cpp
colcon test-result --verbose
ros2 launch goat_control goat_control_system.launch.py
```

These commands build the workspace, load installed packages, run available checks, report results, and launch the control system. Launching control requires configured hardware; inspect `src/goat_control/config/goat_config.yaml` first. For focused C++ work, use `colcon build --packages-up-to goat_control_cpp`.

## Coding Style & Naming Conventions

Use four-space Python indentation and match the existing two-space C++ indentation. Use `snake_case` modules/functions and `PascalCase` classes; retain established ROS topic and message names. C++ targets use C++17. Keep hardware access in motor/IMU utilities and ROS orchestration in nodes. Add brief English comments explaining new variables and methods. Python manifests declare pytest, `ament_flake8`, and `ament_pep257`; C++ uses `ament_lint_auto`. No repository-wide formatter configuration is present.

## Testing Guidelines

No dedicated automated test directory or coverage threshold is currently defined. Add Python regression tests as `test/test_*.py` within the affected package. Run package checks with `colcon test` and inspect results. System-identification scripts named `*_test.py` are hardware experiments; review them before execution. The C++ `imu_io_test` executable is a manual hardware diagnostic.

## Commit & Pull Request Guidelines

History uses informal subjects such as `cpp ongoing` and `torque test`; no consistent prefix convention is established. Prefer concise, imperative subjects identifying the affected component. PRs should explain behavior changes, affected packages, validation commands/results, and hardware/configuration assumptions. Link related issues and include logs or screenshots when relevant.

## Agent Workflow

Before coding, prepare `consult.md` with Conclusion, Logic, and exact Code changes for user approval. Apply `consult.md` feedback when requested. Keep edits scoped; leave Git staging and commits to the user.
