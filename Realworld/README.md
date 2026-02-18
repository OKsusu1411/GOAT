# GOAT Control Architecture 

## Overview
- Core control logic
- Hardware communication layer
- State estimation
- ROS2 interface nodes
- Visualization
- System identification

---

## Package Structure

## goat_control

The control stack is reorganized into a layered architecture under:

```
goat_control/
 ├── core/
 ├── nodes/
 ├── config/
 ├── launch/
```

---

### core/

This directory contains all ROS-independent control and system logic.

#### core/control/

Control algorithms and safety handling:

- `control_pipeline.py` – Main control flow orchestration
- `pd_controller.py` – Joint PD control
- `pi_controller.py` – Wheel PI control
- `safety_limiter.py` – Torque/command limiting logic

#### core/estimation/

State estimation and sensor processing:

- `state_manager.py` – Centralized robot state management
- `filters.py` – Filtering utilities
- `imu.py` – IMU data handling
- `state_types.py` – State data structures

#### core/comm/

Hardware communication abstraction:

- `can.py` – CAN interface
- `motor_driver.py` – Motor driver interface
- `protocol.py` – Low-level protocol definitions

#### core/model/

Robot model abstraction:

- `goat_model.py` – Robot structure definition
- `model_builder.py` – Model construction utilities

The `core` layer contains no ROS dependencies and is fully reusable for simulation, testing, or alternative interfaces.

---

### nodes/

This directory contains ROS2 node wrappers around the core logic.

Nodes are responsible for:

- Topic subscription and publication
- Interfacing with ROS messages
- Calling core modules
- Sending commands to motors

Main nodes include:

- `control_node.py` – Main control execution node
- `state_estimation_node.py` – State estimation ROS wrapper
- `motor_io_node.py` – Motor communication bridge
- `motor_command_node.py` – Command publishing
- `policy_node.py` – Policy input interface
- `policy_keyboard_tester.py` – Manual policy testing
- `log_viewer_node.py` – Motor torque/state logger
- `plot_node.py` – Real-time plotting utility

Nodes are now lightweight and delegate computation to the `core` layer.

---

### config/

- `goat_config.yaml` – Central configuration for control parameters

---

### launch/

- `goat_controller_launch.py`
- `goat_control_system.launch.py`

Launch files are updated to reflect the new modular node structure.

---

## goat_description

This package handles visualization and robot model representation.

Structure:

```
goat_description/
 ├── urdf/
 ├── meshes/
 ├── launch/
```

- `urdf/WF_GOAT.urdf` – Main robot URDF
- Mesh files for all links (base, hip, thigh, calf, wheel)
- `display.launch.py` – RViz visualization launch

This package is dedicated to:

- Publishing `robot_description`
- Displaying joint states and TF
- Visualizing real-world behavior in RViz2

No control logic exists in this package.

---

## goat_sysid

This package contains system identification utilities.

Structure:

```
goat_sysid/
 ├── breakaway_torque_tester.py
 ├── dynamic_friction_id_node.py
 ├── wheel_friction_id_node.py
 ├── ls_joint_version.py
 ├── ls_wheel_version.py
```

Functionality includes:

- Static friction estimation
- Dynamic friction identification
- Breakaway torque testing
- Least-squares based parameter fitting for joints and wheels

This package is fully separated from the main control pipeline.

---

## motor_interfaces

Custom ROS message definitions:

- `MotorStates.msg`
- `BaseStates.msg`

Used for structured communication between nodes.

---