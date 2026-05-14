# solution.md — End-to-End Action Latency Measurement

## Conclusion

**What:** Add timing instrumentation that measures the *total* time from the moment
`controller_node` begins a control cycle (agent computes the action) until
`motor_io_node` finishes writing that action onto the CAN bus.

**Why:** Before sim2real we need to know the real actuation latency of the pipeline.
The policy assumes a fixed 200 Hz (5 ms) loop, but the pipeline crosses two ROS2
processes (`controller_node` → `/commands` → `motor_io_node` → CAN). Currently only
the CAN write time (`dt_command`) is printed in `motor_io_node.py:218`; the agent
compute time and the cross-process transport time are not measured. We need both.

**Result of this change — two numbers get logged:**
1. `controller_internal` (in `controller_node.py`) — time spent inside the controller
   process: agent inference + safety limiter.
2. `e2e` end-to-end latency (in `motor_io_node.py`) — agent start → CAN write done,
   reported as rolling mean / min / max / std plus an effective rate `1/mean`.

---

## Logic

### How to share a clock across two processes

`controller_node` and `motor_io_node` are **separate processes**, so `time.monotonic()`
(used today for `dt_sec` and `dt_command`) is *not* comparable between them — each
process has its own monotonic origin.

The ROS clock (`self.get_clock().now()`) returns system wall-clock time. Both nodes
run with `use_sim_time = False` (explicit in `controller_node.py:46-52`, default in
`motor_io_node`), so their ROS clocks share the same epoch and **are** comparable.

The `JointState` message on `/commands` already carries a `header.stamp` field. We use
it as the carrier for the shared timestamp `T0`.

### Definition of T0 and T1

- **T0** = start of the controller cycle, captured at the top of
  `controller_node._control_loop()`. This is the moment "the agent calculates the
  action". We stamp the outgoing `/commands` message header with **T0** instead of the
  current publish-time stamp.
- **T1** = the instant `motor_io_node._tick()` finishes the CAN write
  (`write_torques_and_read_states` returns) — i.e. the action is on the bus.

`e2e_latency = T1 - T0` covers:
agent inference + safety limiter + `/commands` publish + DDS transport +
wait for the next `motor_io_node` tick + torque clip/LPF/convert + CAN read+write.

### Why the latency includes a "tick wait"

`motor_io_node` writes CAN inside its own 200 Hz timer (`_tick`), **not** inside the
`_on_command` subscription callback. So a freshly received command waits up to one
`motor_io` period before it is sent. That wait is real actuation delay, so it is
correctly included in `e2e_latency`. (If you later want the pure pipeline cost without
this async wait, the alternative is to write CAN directly inside `_on_command` — that
is an architecture change, out of scope here.)

### Frequency vs. latency

The user asked for "frequency", but the meaningful quantity for a pipeline is
**latency** (how long one action takes to reach the bus). We report:
- `e2e` latency stats (mean/min/max/std) — the primary metric.
- `eff_rate = 1 / mean(e2e_latency)` — a convenience figure.
- The actual *throughput* frequency (how often a new command lands on CAN) is still
  the `/commands` publish rate / `motor_io` tick rate, measurable independently with
  `ros2 topic hz /commands`. Latency ≠ 1/throughput because the pipeline is deeper
  than one stage.

### Statistics buffering

`motor_io_node` collects each tick's `e2e_latency` into a list and, every 200 samples
(~1 s at the 200 Hz target), prints aggregate stats and clears the buffer. This avoids
flooding the console at 200 Hz while still giving live numbers.

---

## Code

All line numbers refer to the files **before** editing.

### Change 1 — `controller_node.py`: capture T0 at cycle start

File: `src/goat_control/goat_control/nodes/controller_node.py`

**1a. Initialize the stamp field in `__init__`** (insert after `self.last_tick_time = time.monotonic()` at line 126):

```python
        # Timing
        self.last_tick_time = time.monotonic()

        # ROS-clock timestamp marking the start of the current control cycle (T0).
        # Carried in the /commands header so motor_io_node can measure end-to-end latency.
        self._cycle_start_stamp = self.get_clock().now()
```

**1b. Set T0 at the top of `_control_loop`** (insert inside `_control_loop`, right after `self.last_tick_time = now_time` at line 303):

```python
        self.last_tick_time = now_time

        # T0: start of this control cycle (moment the agent begins computing).
        # ROS clock is used because it is comparable across processes.
        self._cycle_start_stamp = self.get_clock().now()
```

**1c. Log the controller-internal time** (insert in `_control_loop` right after `tau[:] = safe_torque` at line 361, before `self._publish_torque_command(...)`):

```python
        # Publish torque command
        tau[:] = safe_torque

        # Time spent inside controller_node this cycle: agent inference + safety limiter.
        controller_internal_sec = (self.get_clock().now() - self._cycle_start_stamp).nanoseconds * 1e-9
        self.logger.info(
            f"[timing] controller_internal: {controller_internal_sec * 1e3:.3f} ms\r",
            throttle_duration_sec=1.0,
        )

        self._publish_torque_command(q_ref, v_ref, tau)
```

**1d. Stamp the outgoing message with T0 instead of publish-time** — in `_publish_torque_command`, replace line 370:

```python
    def _publish_torque_command(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray) -> None:
        """Publish torque command to /commands topic."""
        msg = JointState()
        # Stamp with cycle-start time T0 (not publish time) so motor_io_node measures
        # latency from the moment the agent began computing this action.
        msg.header.stamp = self._cycle_start_stamp.to_msg()
```

> Note: every `_publish_torque_command` call path (idle / kill-switch / stale / normal)
> already runs after step 1b sets `self._cycle_start_stamp`, so the stamp is always
> valid for the current cycle.

### Change 2 — `motor_io_node.py`: measure T1 and report e2e latency

File: `src/goat_control/goat_control/nodes/motor_io_node.py`

**2a. Import the ROS `Time` helper** (add near the top imports, after line 8 `import rclpy`):

```python
import rclpy
from rclpy.node import Node
from rclpy.time import Time          # for reconstructing a ROS time from a header stamp
```

**2b. Add state buffers in `__init__`** (replace lines 140-142):

```python
        # Latest command buffer
        self._latest_torque_cmd: Optional[np.ndarray] = None
        self._latest_torque_cmd_time_sec: float = 0.0

        # Header stamp (T0) of the latest /commands message — controller cycle start.
        self._latest_torque_cmd_stamp = None
        # Rolling buffer of end-to-end latencies [sec], flushed as stats every N ticks.
        self._e2e_latency_buf: list[float] = []
```

**2c. Capture T0 from the incoming message** in `_on_command` (insert after line 167-168):

```python
        self._latest_torque_cmd = torque
        self._latest_torque_cmd_time_sec = time.time()
        # Keep the controller's cycle-start stamp (T0) for end-to-end latency.
        self._latest_torque_cmd_stamp = msg.header.stamp
```

**2d. Measure T1 and log stats** in `_tick`, right after `t2_command = time.monotonic()` (line 204) and the existing `dt_command` block:

```python
        t2_command = time.monotonic()

        # ... existing JointState publish code stays here ...

        dt_command = t2_command - t1_command
        print(f"dt_command : {dt_command:.4f}")   # CAN read+write portion only

        # --- End-to-end latency: T0 (agent start) -> T1 (CAN write done) ---
        if self._latest_torque_cmd_stamp is not None:
            # T1 now, T0 from the carried header stamp; ROS clock is cross-process safe.
            cmd_stamp = Time.from_msg(self._latest_torque_cmd_stamp)
            e2e_latency_sec = (self.get_clock().now() - cmd_stamp).nanoseconds * 1e-9
            self._e2e_latency_buf.append(e2e_latency_sec)

            # Flush rolling stats once per ~200 ticks (~1 s at the 200 Hz target).
            if len(self._e2e_latency_buf) >= 200:
                latency_arr = np.asarray(self._e2e_latency_buf)
                self.get_logger().info(
                    f"[timing] e2e  mean={latency_arr.mean() * 1e3:.3f}ms "
                    f"min={latency_arr.min() * 1e3:.3f}ms "
                    f"max={latency_arr.max() * 1e3:.3f}ms "
                    f"std={latency_arr.std() * 1e3:.3f}ms "
                    f"eff_rate={1.0 / latency_arr.mean():.1f}Hz"
                )
                self._e2e_latency_buf.clear()
```

> Place this block *after* the `js` JointState publish so the measurement does not
> delay the state publication. `dt_command` (CAN-only) and `e2e` (full pipeline) are
> complementary: `e2e - dt_command` ≈ transport + tick-wait + torque processing.

### How to run and read the result

1. Rebuild: `colcon build --packages-select goat_control && source install/setup.bash`
2. Launch as usual: `ros2 launch goat_control goat_control_system.launch.py`
3. In the `controller_node` terminal, press `p` or `n` to enter a control mode
   (no `/commands` are published while idle, so no timing prints until then).
4. Read the logs:
   - `controller_node` terminal → `[timing] controller_internal: X.XXX ms` (once/sec)
   - `motor_io_node` terminal → `[timing] e2e mean=... eff_rate=...Hz` (once/sec)
5. Optional cross-check of throughput frequency:
   `ros2 topic hz /commands`

### Interpretation checklist for sim2real

- `controller_internal` should be well under 5 ms; if not, the agent inference or
  safety limiter is the bottleneck.
- `e2e mean` is the real actuation delay the trained policy is *not* aware of.
  Compare it against the sim control period (5 ms). A large gap (e.g. e2e ≈ 8-12 ms)
  is a genuine sim-to-real discrepancy to compensate or retrain against.
- `e2e - controller_internal - dt_command` isolates DDS transport + the
  `motor_io` tick-wait; if this dominates, consider writing CAN directly in
  `_on_command` instead of the timer `_tick`.
