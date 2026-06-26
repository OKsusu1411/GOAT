# solution.md — Restore synchronous 200 Hz joint observation

## Conclusion

The three symptoms (position ≠ 0° after calibration, huge CMD_POS / CMD_TAU,
±1080° spikes at certain joint positions) all come from the same regression
introduced by `5a6db0c freq amplify`: the control hot-path was switched from
**synchronous TX/RX per tick** to **fire-and-forget TX + cached "latest
frame" RX**.

A 200 Hz feedback controller with single-turn → multi-turn encoder
integration depends on a **constant Δt between successive observations**.
The cache violates that invariant — tick-to-tick staleness is variable —
which means:

- the wrap detector cannot tell "real wrap" from "stale cache catching up"
  (root cause of the ±1080° spikes),
- the very first hot-path tick differences a cached 0xA1 reply against an
  init 0x9C count taken ~ms earlier on a different command (root cause of
  "position ≠ 0° after calibration"),
- a momentarily stalled reader thread leaves velocity and position frozen
  for several ticks — unsafe for a balance controller.

The right fix is **not** the speed-aware wrap detector + 1 Hz re-anchor I
previously proposed in `solution.md` — those were band-aids. The right fix
is to put the hot path back on synchronous TX/RX (1 ms typical, 2 ms
worst-case on 1 Mbps CAN), while keeping the parts of `freq amplify` that
were actually right (move 0x9A error-flag poll off the hot path, reuse a
persistent thread pool, keep both buses busy in parallel).

We change one design choice: replace the "latest frame" cache with **per-key
`threading.Event` dispatch**, so the hot path can fire all 8 0xA1 commands,
then wait for *this tick's* 8 replies with a tight shared deadline. Each
reply is consumed exactly once. Staleness is impossible by construction.

---

## Logic

### What `freq amplify` got right vs. what it broke

`5a6db0c` made three changes; the first two are correct and we keep them.

| Change in `freq amplify`                              | Verdict       |
|--------------------------------------------------------|---------------|
| Move 0x9A (error-flag) poll off the hot path → 1 Hz timer | ✅ keep       |
| Reuse persistent `_io_pool` instead of spawning threads/tick | ✅ keep       |
| Replace blocking `txrx()` with `send_only()` + cached `latest_state2()` | ❌ revert     |

The third change is the cause of all three reported symptoms. We undo only
that one, and only on the hot path.

### Bandwidth budget confirms sync is feasible

On 1 Mbps CAN with standard 8-byte frames:

- One frame on the wire = ~111 µs
- TX(0xA1) + RX(reply) per motor = ~222 µs
- 4 motors on one bus, serialized = ~0.9 ms
- Two buses in parallel (separate SocketCAN sockets / kernel rings) = **~0.9 ms total per tick**

A 200 Hz control tick has 5 ms of budget. CAN consumes <1 ms typical, <2 ms
worst case (with one motor briefly silent). Margin is comfortable. The OLD
code (pre-`freq amplify`) was already achieving this — its only real
problem was the periodic 4-5 ms spike from running 0x9A every 10th tick,
and that fix is independently preserved.

### Why per-key `Event` dispatch beats "latest frame" cache

The reader thread is already draining the bus into a dict
(`latest_rx_frames_by_key`). The problem is not the reader thread — it's
that the hot path *reads from* a "give me whatever you have right now"
cache. We want "give me **this tick's** reply, or `None` after deadline."

Add a parallel dict of `threading.Event` keyed the same way
`((arb_id, cmd_byte))`. The reader thread, when it caches a frame, also
**sets** the corresponding event. Hot-path callers **clear** the events
they care about before sending the request, then **wait** on them.

Properties this gives us, by construction:

- Reply consumed exactly once: cleared → set → consumed → cleared next tick.
- Tick-bounded latency: per-tick deadline of ~3 ms total RX wait, shared
  across all 8 motors. Pipeline keeps both buses busy.
- Failsafe on motor silence: deadline elapses → driver returns `None` →
  we inject `NaN` into the joint state slot → existing
  `controller_node._sensor_data_has_nan()` kill switch fires.
- The reader thread keeps its role as the sole `bus.recv()` owner —
  unchanged for the 1 Hz 0x9A slow-poll, which is already happy with the
  cache (1-second staleness is fine for safety telemetry).

### Why this kills all three symptoms

| Symptom | How sync fixes it |
|---|---|
| **(3) ±1080° spikes** | Δt between successive observations is back to a constant 5 ms. Max realistic motor speed (≈10 rev/s) produces ≤820 counts/tick, well under the 8192 half-range threshold. The original wrap detector works as written; no speed-aware logic needed. |
| **(1) Position ≠ 0° after calibration** | The init multi-turn anchor (0x92) and the first hot-path encoder count (0xA1) are now both fresh, deterministic reads on a stationary robot. With Change 4 below pairing 0x92 ↔ 0x9C tightly at init, the first tick produces `delta_count = 0` exactly and `motor_angle_deg == anchor_angle_deg` exactly. After rebuild + restart, the same anchor reproduces, so `joint_state.position == joint_offsets`, so published position = 0. |
| **(2) Huge CMD_POS / CMD_TAU** | Was a downstream consequence of (3) and (1). With clean position, `NominalController` builds `q_ref_traj` from a clean start. |

### What we delete from the prior solution.md

All three patches from the previous version become unnecessary and are
removed: the speed-aware wrap detector, the 1 Hz 0x92 re-anchor, and the
"skip first delta" hack. They were treating symptoms of the broken
invariant; the invariant is now restored at the source.

### One trade-off worth naming

Sync means the slowest motor sets the tick deadline. If one motor is
intermittently silent (cable, EMI, brownout), the tick blocks until its
per-motor deadline elapses (~2 ms). At a 5 ms tick budget this is fine —
and the OLD code already had this property. The cache approach traded
correctness for jitter immunity to a single slow motor; for a balance
controller that was the wrong trade.

---

## Code

### Change 1 — Add per-key event dispatch to `CanInterface`

**File:** `src/goat_control/goat_control/utils/motor/can.py`

**1a.** Add an events dict next to the existing frame cache (in `__init__`,
right after `self._rx_first_keys_seen` declaration at can.py:58):

```python
# Per-key arrival events for synchronous hot-path dispatch. Hot-path
# callers clear the event before sending a request, then wait on it for
# THIS tick's reply. The reader thread sets the event when a frame for
# that key is cached. Decouples the hot path from the slow-poll path
# which still uses the cache directly.
self.frame_events: dict[tuple[int, int], threading.Event] = {}
self._events_lock = threading.Lock()
```

**1b.** Add an `event_for_key` helper after `get_latest_frame`
(can.py:150):

```python
def event_for_key(self, arbitration_id: int, cmd_byte: int) -> threading.Event:
    """Return (creating if needed) the arrival Event for one reply key.
    Lazily allocated so we only carry events for keys the hot path uses."""
    key = (arbitration_id, cmd_byte)
    with self._events_lock:
        ev = self.frame_events.get(key)
        if ev is None:
            ev = threading.Event()
            self.frame_events[key] = ev
        return ev
```

**1c.** Make the reader thread set the event whenever it caches a frame.
**Modify** the body of `_rx_loop` (can.py:119-145), replacing the cache
write section:

```python
key = (msg.arbitration_id, msg.data[0])
with self._rx_lock:
    self.latest_rx_frames_by_key[key] = msg
    self.rx_frame_count += 1
# Wake any hot-path waiter on this exact key. Cheap no-op if no waiter
# was ever registered for this key.
ev = self.frame_events.get(key)
if ev is not None:
    ev.set()
```

### Change 2 — Add a synchronous `await_state2` to `MotorDriver`

**File:** `src/goat_control/goat_control/utils/motor/motor_driver.py`
**Insert after motor_driver.py:112** (alongside the existing
`latest_state2`). Keep `send_torque_only` and `latest_state2` — the
manager will compose them into a single sync call.

```python
def clear_state2_event(self) -> None:
    """Arm this motor for a fresh 0xA1 reply.

    Must be called BEFORE send_torque_only() each tick so the subsequent
    wait blocks until THIS tick's reply lands (not the previous one's)."""
    self.can_interface.event_for_key(self.can_ids.rx_id, 0xA1).clear()
    self.can_interface.event_for_key(self.can_ids.tx_id, 0xA1).clear()

def await_state2(self, deadline_monotonic: float):
    """Block until a fresh 0xA1 reply arrives for this motor, or the
    shared deadline elapses. Returns the can.Message or None on timeout.

    `deadline_monotonic` is an absolute time.monotonic() value shared by
    all 8 motors this tick — keeps total RX wait bounded regardless of
    motor count."""
    import time
    rx_ev = self.can_interface.event_for_key(self.can_ids.rx_id, 0xA1)
    remaining = max(0.0, deadline_monotonic - time.monotonic())
    if rx_ev.wait(remaining):
        msg = self.can_interface.get_latest_frame(self.can_ids.rx_id, 0xA1)
        if msg is not None:
            return msg
    # rx_id timed out — try tx_id fallback (mirrors txrx's accept_tx_echo_diff).
    return self.can_interface.get_latest_frame(self.can_ids.tx_id, 0xA1)
```

### Change 3 — Rewrite `write_torques_and_read_states` as sync pipeline

**File:** `src/goat_control/goat_control/utils/motor/motor_manager.py`
**Lines:** 566-618 (replace the entire method body)

```python
def write_torques_and_read_states(
    self,
    current_cmd_amp: Sequence[float],
    timeout: float = 0.003,         # total RX deadline shared across all 8 motors
    perform_slow_poll: bool = False,  # kept for API compatibility; ignored
) -> MotorStatesData:
    """Synchronous fire-all-then-wait-all torque + state2 pass.

    Phase 1 — clear each motor's 0xA1 reply event, then send its torque
              command. TXs are non-blocking (kernel ring); both buses run
              in parallel by virtue of separate CanInterface instances.
    Phase 2 — for each motor, wait on its arrival event until the shared
              deadline. On timeout, inject NaN so the kill switch fires
              instead of letting stale data drive the controller.

    Δt between successive ticks is now constant (the control timer period),
    so the encoder wrap detector in _update_motor_angle_from_encoder works
    as originally designed without any speed-sign disambiguation.
    """
    # Phase 1 — arm + fire. Sequential is fine: bus.send() goes into the
    # kernel TX ring without blocking on the wire.
    t_submit = time.perf_counter()                                       # [timing]
    for motor_index, amp in enumerate(current_cmd_amp):
        driver = self.motor_drivers[motor_index]
        driver.clear_state2_event()
        driver.send_torque_only(float(amp))
    t_fired = time.perf_counter()                                        # [timing]

    # Phase 2 — bounded wait, one shared deadline so total RX time is
    # bounded by `timeout` rather than 8 × per-motor timeout.
    deadline_monotonic = time.monotonic() + float(timeout)
    for motor_index in range(self.motor_count):
        driver = self.motor_drivers[motor_index]
        response_message = driver.await_state2(deadline_monotonic)
        if response_message is None:
            # Motor silent this tick. Mark its slot NaN so
            # controller_node._sensor_data_has_nan() trips the kill switch
            # — far safer than re-using stale state for a balance loop.
            self.motor_speed_deg_per_sec[motor_index] = float("nan")
            self.motor_phase_current_amp[motor_index] = float("nan")
            continue

        response_data = response_message.data

        # 0xA1 reply has the same byte layout as 0x9C state2.
        self.motor_temperature_c[motor_index] = float(struct.unpack("<b", response_data[1:2])[0])

        motor_current_raw_lsb = struct.unpack("<h", response_data[2:4])[0]
        self.motor_phase_current_amp[motor_index] = float(motor_current_raw_lsb) * self.motor_current_amp_per_lsb

        speed_raw_lsb = struct.unpack("<h", response_data[4:6])[0]
        self.motor_speed_deg_per_sec[motor_index] = float(speed_raw_lsb) * self.speed_deg_per_sec_per_lsb

        self.motor_encoder_count[motor_index] = int(struct.unpack("<H", response_data[6:8])[0])

        self._update_motor_angle_from_encoder(motor_index)
    t_done = time.perf_counter()                                         # [timing]

    # Surface timings (read by controller_node timing log).
    self._last_tx_submit_ms = (t_fired - t_submit) * 1e3
    self._last_tx_wait_ms = (t_done - t_fired) * 1e3

    return self._package_motor_states()
```

### Change 4 — Tighten anchor pairing at init

**File:** `src/goat_control/goat_control/utils/motor/motor_manager.py`

The init `fetch_motor_data` currently reads in order: state2 → state1 →
multi-turn. The anchor then pairs encoder_count (from state2, oldest) with
motor_angle_deg (from multi-turn, newest), with ~ms between them. Reverse
the order so multi-turn and state2 are adjacent.

**Replace motor_manager.py:541-545** (`fetch_motor_data` body in
`decode_motor_encoder`):

```python
def fetch_motor_data(motor_index: int):
    if perform_slow_poll:
        self.poll_state1(motor_index)                # 0x9A (oldest, doesn't feed anchor)
        self.poll_single_or_multi_turn(motor_index)  # 0x92 (feeds anchor_motor_angle_deg)
    self.poll_state2(motor_index)                    # 0x9C (feeds anchor_encoder_count) — last so it pairs tightly with 0x92
```

Now `anchor_motor_angle_deg` and `anchor_encoder_count` are captured ~200 µs
apart on the same bus, instead of ~1-3 ms. On a held-still robot the
remaining mismatch is sub-count; the very first hot-path tick produces
`delta_count = 0` exactly, and the published joint position is `−joint_offset`
exactly, which (after calibration writes that same value as the offset) is
zero.

### Change 5 — Revert the band-aids from the previous solution.md

If any of Changes 1/2/3 from the prior version of this file were already
applied, undo them now — they are no longer needed and add cost without
benefit once the sync invariant is restored.

Specifically remove, from
`src/goat_control/goat_control/utils/motor/motor_manager.py`:

- the speed-sign branches inside `_update_motor_angle_from_encoder` (return
  to the original `|delta_count| > half_range` form),
- `reanchor_from_multi_turn()`,
- `motor_first_encoder_update_skipped` and the "skip first delta" block.

And from `src/goat_control/goat_control/utils/motor/motor_driver.py`:

- `send_multi_turn_request()` and `latest_multi_turn()` (no caller).

And in `src/goat_control/goat_control/nodes/motor_io.py`,
`poll_error_flags_once()` returns to its pre-band-aid body:

```python
def poll_error_flags_once(self) -> None:
    """Read 0x9A error flags on every motor — call from a ~1 Hz timer.
    After Step 3 poll_state1 is fire-and-forget; no need for the thread
    pool. Sequential send_only is essentially free (<1 ms total)."""
    mm = self.motor_manager
    for motor_index in range(mm.motor_count):
        mm.poll_state1(motor_index)
```

(The 1 Hz `error_flag_timer` in `controller_node.py:174` stays as-is.)

---

## Verification

1. `colcon build` and `ros2 launch goat_control goat_control_system.launch.py`.
2. Watch the existing `[timing]` line. Expect `can:` to read **~1.0-1.5 ms**
   (tx ~0.2 / rx ~0.8-1.3) at 200 Hz. If it goes >3 ms consistently, one
   bus is dropping replies — investigate motor health, not the code.
3. Hold the robot in the calibration pose. `ros2 topic echo /joint_states`:
   every `position` value should print within ±0.01 rad of 0 immediately.
4. Power-cycle the robot, repeat step 3 without re-calibrating. Position
   should still be near 0 (proves the anchor reproduces deterministically).
5. Rotate one leg joint by hand through ≥360° motor at a normal pace.
   Position should track smoothly. No ±360° / ±1080° steps anywhere on
   the trajectory.
6. Spin a wheel by hand for ≥5 s — position should accumulate monotonically.
7. Pull a CAN cable on one motor mid-run. Expect: within ~2-3 ms,
   `_sensor_data_has_nan()` trips the kill switch, controller goes to
   zero torque, log prints "NaN detected in joint/IMU state". Reconnect,
   press `r` for manual reset, control resumes cleanly.

Tell me when you've reviewed this and I'll apply Changes 1–5 in order.
