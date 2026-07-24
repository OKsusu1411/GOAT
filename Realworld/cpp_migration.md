# `goat_control` → C++ Migration Plan (`goat_control_cpp`)

## Decisions (locked)

- **Target package:** new `goat_control_cpp` alongside the existing Python `goat_control` (kept until cutover). Cleaner git history and rollback.
- **Boundary:** migrate the **full hot path** to C++ (`controller_node` + `MotorIO` + `ImuIO` + policy) so I/O stays owned **in-process** — no re-introduced DDS topic hops. Side nodes migrate last.
- **First slice (branch `59-imu-io-cpp-migration`):** the IMU stack.

---

## What `goat_control` is today

An `ament_python` ROS2 package. The real architecture is **one hot control node that owns its I/O in-process** — the old `/imu`, `/joint_states`, and `/commands` topic hops were deliberately deleted to kill cross-process DDS latency.

```
controller_node (200 Hz timer loop, ROS2 Node)
├── owns MotorIO  ── utils/motor       (SocketCAN, protocol, drivers, filters)
├── owns ImuIO    ── utils/imu         (serial reader thread + quaternion math)
└── uses          ── utils/controller  (policy [ONNX], nominal, safety)
+ side nodes: calibration_node, log_viewer_node, sim_controller_node, topic_converter_node
```

### IMU slice (this branch)
- `nodes/imu_io.py` — thin adapter; per tick `read_imu()` → `ImuState` msg.
- `utils/imu/imu_manager.py` — `ImuSerialReader` (background thread; line protocol
  `*w,x,y,z,gx,gy,gz,vx,vy,vz,mx,my,mz,t_ms`), `ImuConfig`, `ImuPacket`;
  applies quaternion offset + **deg→rad on gyro**.
- `utils/imu/quaternion_utils.py` — `inverse/multiply/rotate/axis_angle`
  (also used by `calibration_node`).

---

## Dependency translation

| Python | C++ |
|---|---|
| `rclpy` | `rclcpp` |
| `pyserial` | termios (custom `SerialPort` RAII wrapper) |
| `numpy` / quats | **Eigen3** |
| `yaml.safe_load` | **yaml-cpp** |
| `torch` + `.onnx` checkpoints | **ONNX Runtime C++** (inference already uses `ort.InferenceSession`) |
| python-can / socketcan | native SocketCAN (`<linux/can.h>`) |
| `threading` | `std::thread` / `std::mutex` / `std::atomic` |
| `goat_api.msg` | already generates C++ types |
| `ament_python` | **`ament_cmake`** |

**Note:** inference already runs on `onnxruntime`; the `torch` dep and `.pt` glob in `setup.py` are vestigial and will be dropped.

---

## Proposed package layout

```
goat_control_cpp/
├── package.xml                         # build_type: ament_cmake
├── CMakeLists.txt
├── include/goat_control_cpp/
│   ├── imu/
│   │   ├── quaternion_utils.hpp        # Eigen-based; inverse/multiply/rotate/axis_angle
│   │   ├── imu_packet.hpp              # struct ImuPacket
│   │   ├── imu_config.hpp              # struct ImuConfig
│   │   ├── serial_port.hpp             # RAII termios wrapper (replaces pyserial)
│   │   ├── imu_serial_reader.hpp       # background std::thread + mutex + latest vector
│   │   └── imu_io.hpp                  # read_imu() -> ImuState
│   ├── motor/    {can_interface, protocol, motor_driver, motor_manager, filters, motor_io}.hpp
│   └── controller/ {base, policy, fixed_policy, movable_policy, nominal, safety_limiter}.hpp
├── src/
│   ├── imu/*.cpp   motor/*.cpp   controller/*.cpp     # → libgoat_control_core
│   └── nodes/
│       ├── controller_node.cpp   (main executable)
│       ├── calibration_node.cpp   log_viewer_node.cpp
│       ├── sim_controller_node.cpp topic_converter_node.cpp
├── config/  launch/  urdf/  checkpoint/               # copied from python pkg
└── test/                                              # gtest, mirrors ament tests
```

Build a `goat_control_core` library (the old `utils/`) + thin node executables
(the old `nodes/`). Preserves the current `utils/` vs `nodes/` separation exactly.

### CMake dependencies
`rclcpp`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `goat_api`,
`tf2_ros`, `message_filters`, `Eigen3`, `yaml-cpp`, and `onnxruntime`
(find via a small `Findonnxruntime.cmake` or `ament_vendor` — it is **not** a
standard rosdep key; this is the one non-standard dependency).

---

## Phasing (each phase compiles, links, and is testable)

1. **IMU (this branch)** — `quaternion_utils` → `serial_port` → `imu_serial_reader`
   → `imu_io`. Standalone gtest against a recorded serial capture / pty.
   No ROS needed to unit-test the decoder.
2. **Controller node skeleton** — `rclcpp` node, 200 Hz wall timer, params,
   yaml-cpp config load, owns `ImuIO`, publishes `/imu` for logging.
   First runnable C++ executable.
3. **Motor** — SocketCAN via `<linux/can.h>` (cleaner than python-can),
   `protocol`, `motor_driver`, `motor_manager`, `filters`, `motor_io`;
   wire into the loop.
4. **Controllers** — `safety_limiter`, `nominal_controller`, then
   `policy_controller` on ONNX Runtime (fixed + movable). Full closed loop.
5. **Side nodes** — calibration / log_viewer / sim / topic_converter,
   then cut launch files over and retire the Python package.

---

## IMU slice — concrete design (Phase 1)

- **`quaternion_utils`** — free functions on `Eigen::Quaterniond` / `Eigen::Vector3d`.
  The Python versions are overloaded to accept both raw `[w,x,y,z]` and
  `ImuState`-like objects; these split into clean typed overloads.
  Shared by `imu_serial_reader` (offset) and later `calibration_node`.
- **`SerialPort`** — RAII termios wrapper: open/configure baud/`readline()`.
  Replaces pyserial with no external dependency.
- **`ImuSerialReader`** — `std::thread` read loop, `std::mutex`-guarded
  `latest_raw_vector` (14 floats), `get_latest_packet()` decodes: apply
  `imu_offsets` quat multiply, **deg→rad on gyro**, parse `*`-prefixed CSV.
  `ImuConfig` carries port/baud/timeout + `imu_offsets` from yaml-cpp.
- **`ImuIO`** — owns the reader, `read_imu()` fills a
  `goat_api::msg::ImuState`, caches `latest_imu_state`.
  Same surface the C++ `controller_node` calls per tick.

---

## Resolved / noted

- **Inference:** ONNX Runtime C++ — direct map, no LibTorch. Drop `torch` from deps.
- **Config keys** (`imu_offsets`, `num_joints`, `can_channels`, `motor_bus_index`,
  `motor_node_ids`, …) read via yaml-cpp from the same `goat_config.yaml`.
- **Real-time:** start with `create_wall_timer` + single-thread spin (matches today);
  can promote to a dedicated `SCHED_FIFO` thread later if 200 Hz jitter appears.
- **Keyboard mode switch** in `controller_node` (raw termios via `tty`/`termios`)
  ports directly to C++ termios.
