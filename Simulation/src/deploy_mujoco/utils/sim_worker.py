"""Physics thread + the locks that keep it separate from the ROS threads.

The MuJoCo integration runs in its own thread here, so ROS callbacks never
step physics and physics never blocks on DDS. The two sides meet at exactly
three small critical sections, each guarded by its own lock:

    ROS cmd callback --(_cmd_lock)--> latest command --+
                                                       |
                                             sim thread (_data_lock: MjData,
                                             set_ctrl / step / snapshot / sync)
                                                       |
    ROS publish timer <--(_snap_lock)-- latest snapshot +

Rules that make this deadlock-free and tear-free (see solution.md):

1. ``_cmd_lock`` and ``_snap_lock`` are leaves: nothing else is acquired while
   holding them, and they only ever guard a reference swap.
2. ``_data_lock`` is the outermost lock. The only lock taken under it is
   ``MujocoSim._lock`` (the viewer key-flag lock), which is itself a leaf.
3. Nothing that can block on I/O (DDS publish, logging of big payloads, sleep)
   happens while any lock is held.

This module has no ROS dependency, so the locking can be exercised directly in
tests with nothing but MuJoCo.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .mujoco_sim import MujocoSim, SimSnapshot

logger = logging.getLogger(__name__)


class SimWorker:
    """Runs ``MujocoSim`` in a dedicated thread behind a data lock.

    Producer side (any ROS thread): :meth:`submit_ctrl`.
    Consumer side (any ROS thread): :meth:`latest_snapshot`.
    Neither blocks for longer than a reference swap.
    """

    def __init__(self,
                 sim: MujocoSim,
                 steps_per_cmd: int = 1,
                 use_viewer: bool = False,
                 realtime: bool = True,
                 cmd_timeout: float = 0.5) -> None:
        self.sim = sim
        self.steps_per_cmd = max(1, int(steps_per_cmd))
        self.use_viewer = bool(use_viewer)
        self.realtime = bool(realtime)
        self.cmd_timeout = float(cmd_timeout)
        self.control_period = self.steps_per_cmd * sim.timestep

        # --- locks (see module docstring for the ordering rules) --------- #
        self._data_lock = threading.Lock()   # guards MjData: ctrl/step/snapshot/sync
        self._cmd_lock = threading.Lock()    # guards the latest-command box
        self._snap_lock = threading.Lock()   # guards the latest-snapshot box

        # --- shared state, each field owned by the lock above it --------- #
        self._cmd_ctrl: Optional[np.ndarray] = None
        self._cmd_stamp = 0.0
        self._cmd_seq = 0            # commands accepted from ROS
        self._cmd_used = 0           # distinct commands a tick actually applied
        self._cmd_repeat = 0         # ticks that re-applied the previous command
        self._cmd_last_used = 0      # seq of the command applied last tick

        self._snap: Optional[SimSnapshot] = None
        self._snap_seq = 0

        # --- thread lifecycle (Event is itself thread-safe) -------------- #
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._finished = threading.Event()
        self._exit_reason = ''
        self._error: Optional[BaseException] = None

        # Tick counters live under _snap_lock and are updated in the same
        # critical section as the snapshot, so a reader sees stats and state
        # from the same tick. _warned_stale is sim-thread-private.
        self._ticks = 0
        self._stale_ticks = 0
        self._warned_stale = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError('SimWorker already started')
        self._thread = threading.Thread(target=self._run, name='mujoco_sim', daemon=True)
        self._thread.start()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until the first snapshot exists (or the thread died trying)."""
        while not self._ready.wait(0.05):
            if self._finished.is_set():
                return False
            timeout -= 0.05
            if timeout <= 0.0:
                return False
        return True

    def stop(self, reason: str = 'stop requested') -> None:
        """Ask the sim thread to finish the current tick and exit."""
        if not self._exit_reason:
            self._exit_reason = reason
        self._stop.set()

    def join(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    @property
    def exit_reason(self) -> str:
        return self._exit_reason

    @property
    def error(self) -> Optional[BaseException]:
        """Exception that killed the sim thread, if any."""
        return self._error

    # ------------------------------------------------------------------ #
    # Producer: ROS command callback -> sim thread
    # ------------------------------------------------------------------ #
    def submit_ctrl(self, ctrl: np.ndarray, stamp: Optional[float] = None) -> int:
        """Store a full-length ctrl vector as the latest command.

        Called from a ROS executor thread. Newest wins: if commands arrive
        faster than the sim ticks, the intermediate ones are dropped rather
        than queued, so the sim never works through a stale backlog. The copy
        is made before the lock so the critical section is a pointer swap.
        """
        vec = np.array(ctrl, dtype=float, copy=True)
        if vec.shape != (self.sim.nu,):
            raise ValueError(f'ctrl length {vec.shape} != nu ({self.sim.nu})')
        now = time.monotonic() if stamp is None else stamp
        with self._cmd_lock:
            self._cmd_ctrl = vec
            self._cmd_stamp = now
            self._cmd_seq += 1
            return self._cmd_seq

    # ------------------------------------------------------------------ #
    # Consumer: sim thread -> ROS publish timer
    # ------------------------------------------------------------------ #
    def latest_snapshot(self) -> Tuple[int, Optional[SimSnapshot]]:
        """Return ``(seq, snapshot)`` of the most recent completed tick.

        Called from a ROS executor thread. The snapshot is immutable and fully
        detached from MjData, so the caller can build and publish messages from
        it with no lock held. ``seq`` increments once per tick; an unchanged
        seq means physics has not advanced since the last read.
        """
        with self._snap_lock:
            return self._snap_seq, self._snap

    @property
    def stats(self) -> dict:
        """Throughput counters, for spotting drift between the two loops.

        ``cmd_received - cmd_used`` is how many commands were dropped because
        the controller outran the sim; ``cmd_repeat`` is how many ticks reused
        a command because the sim outran the controller. Both are expected to
        be non-zero in a free-running setup -- they are a rate mismatch, not an
        error.
        """
        with self._cmd_lock:
            received, used, repeat = self._cmd_seq, self._cmd_used, self._cmd_repeat
        with self._snap_lock:
            ticks, stale = self._ticks, self._stale_ticks
        return {
            'ticks': ticks,
            'cmd_received': received,
            'cmd_used': used,
            'cmd_repeat': repeat,
            'stale_ticks': stale,
        }

    # ------------------------------------------------------------------ #
    # The sim thread itself
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            # The viewer is opened, synced and closed on this one thread so no
            # GL call ever crosses a thread boundary.
            if self.use_viewer:
                self.sim.open_viewer()

            self._publish_snapshot()   # seed: state before the first step
            self._ready.set()

            next_wall = time.monotonic()
            while not self._stop.is_set():
                if self.sim.is_quit_requested:
                    self.stop('quit requested from viewer')
                    break
                if self.use_viewer and not self.sim.is_viewer_running:
                    self.stop('viewer closed')
                    break

                self._tick()

                next_wall += self.control_period
                if self.realtime:
                    remaining = next_wall - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    elif remaining < -10 * self.control_period:
                        # Fell far behind (a long GC pause, a slow render):
                        # resync instead of sprinting to catch up.
                        next_wall = time.monotonic()
        except BaseException as exc:  # noqa: BLE001 - reported to the node
            self._error = exc
            self.stop(f'sim thread error: {exc!r}')
            logger.exception('sim thread died')
        finally:
            try:
                self.sim.close_viewer()
            finally:
                self._ready.set()      # never leave wait_ready() hanging
                self._finished.set()

    def _tick(self) -> None:
        """One control tick: take command -> advance -> hand off snapshot."""
        # 1) command box (leaf lock, no physics inside)
        now = time.monotonic()
        with self._cmd_lock:
            ctrl = self._cmd_ctrl
            seq = self._cmd_seq
            age = now - self._cmd_stamp
            fresh = ctrl is not None and (self.cmd_timeout <= 0.0 or age <= self.cmd_timeout)
            if fresh:
                if seq != self._cmd_last_used:
                    self._cmd_used += 1
                    self._cmd_last_used = seq
                else:
                    self._cmd_repeat += 1

        if not fresh:
            ctrl = np.zeros(self.sim.nu)

        # 2) physics (data lock: the only place MjData is touched, and the
        #    only lock taken under it is MujocoSim's leaf flag lock)
        with self._data_lock:
            did_reset = self.sim.consume_reset_request()
            if did_reset:
                # A reset is its own tick: no ctrl, no step. That leaves the
                # reset state visible to readers for a full control period (a
                # reset folded into a stepping tick would be published already
                # one step old, with the pre-reset command applied to it).
                self.sim.reset()
            else:
                self.sim.set_ctrl(ctrl)
                if not self.sim.is_paused:
                    self.sim.step(self.steps_per_cmd)
            snap = self.sim.snapshot()
            if self.use_viewer:
                self.sim.sync()

        # 3) snapshot box (leaf lock, no message building inside)
        self._store_snapshot(snap, tick=True, stale=not fresh and not did_reset)
        if did_reset:
            return

        if not fresh and not self._warned_stale and seq > 0:
            self._warned_stale = True
            logger.warning('no command within %.3fs; holding ctrl at zero',
                           self.cmd_timeout)
        elif fresh:
            self._warned_stale = False

    def _publish_snapshot(self) -> None:
        with self._data_lock:
            snap = self.sim.snapshot()
        self._store_snapshot(snap)

    def _store_snapshot(self, snap: SimSnapshot,
                        tick: bool = False, stale: bool = False) -> None:
        with self._snap_lock:
            self._snap = snap
            self._snap_seq += 1
            if tick:
                self._ticks += 1
            if stale:
                self._stale_ticks += 1

    # ------------------------------------------------------------------ #
    # Locked access for anything the node needs beyond the snapshot
    # ------------------------------------------------------------------ #
    def request_reset(self) -> None:
        """Reset from a ROS thread, applied at the top of the next tick.

        Goes through the same flag MujocoSim's ``r`` key uses, so the reset
        still happens on the sim thread and never races a step in flight.
        """
        self.sim.request_reset()

    def data_lock(self):
        """Context manager for code that must touch MjData from a ROS thread.

        Use sparingly: everything held here stalls physics.
        """
        return self._data_lock
