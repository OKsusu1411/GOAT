#!/usr/bin/env python3
"""Verification for the SimWorker locking. Needs MuJoCo only -- no ROS.

    python3 tools/test_sim_worker.py

Covers the claims solution.md makes: no torn snapshots under concurrent
readers, no exceptions with several ROS-side threads hammering the mailboxes,
newest-command-wins accounting, reset landing between steps, the command
timeout falling back to zero ctrl, and realtime pacing.
"""
import logging, sys, threading, time
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(name)s: %(message)s')

from utils.mujoco_sim import MujocoSim, SimConfig
from utils.sim_worker import SimWorker

MODEL = str(DEPLOY / 'config' / 'xml' / 'goat_floating.xml')
DT, N = 0.005, 1

def make_sim():
    sim = MujocoSim(SimConfig(model_path=MODEL, home_keyframe='home',
                              timestep=DT, inspect_on_load=False))
    sim.reset()          # what GoatMujocoNode does before handing it to the worker
    return sim

fails = []
def check(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {extra}' if extra else ''))
    if not cond:
        fails.append(name)

# ---------------------------------------------------------------- test 1
# Torn-read: a concurrently-read snapshot must equal a single-threaded replay
# at the same sim time (a snapshot mixing two instants would not).
w = SimWorker(make_sim(), steps_per_cmd=N, use_viewer=False, realtime=False, cmd_timeout=0.0)
w.start(); assert w.wait_ready(), w.error
seen = {}
stop = threading.Event()
def reader():
    while not stop.is_set():
        seq, s = w.latest_snapshot()
        if s is not None:
            seen[round(s.time, 9)] = (s.qpos.copy(), s.qvel.copy(), s.base_pos.copy())
readers = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
[t.start() for t in readers]
sys.setswitchinterval(1e-6)          # maximize thread interleaving
time.sleep(1.0)
stop.set(); [t.join() for t in readers]
w.stop('test done'); w.join(); sys.setswitchinterval(0.005)

ref = make_sim(); ref.reset()
mismatch, compared = [], 0
for _ in range(4000):
    t = round(ref.data.time, 9)
    if t in seen:
        compared += 1
        q, v, b = seen[t]
        if not (np.allclose(q, ref.snapshot().qpos, atol=0, rtol=0)
                and np.allclose(v, ref.snapshot().qvel, atol=0, rtol=0)):
            mismatch.append(t)
    ref.set_ctrl(np.zeros(ref.nu)); ref.step(N)
check('snapshot matches single-threaded replay bit-for-bit',
      compared > 50 and not mismatch, f'compared={compared} mismatch={len(mismatch)}')
check('worker exited cleanly', w.error is None, f'reason={w.exit_reason!r}')

# ---------------------------------------------------------------- test 2
# Concurrent producers + consumers under an aggressive switch interval.
w = SimWorker(make_sim(), steps_per_cmd=N, use_viewer=False, realtime=False, cmd_timeout=0.5)
w.start(); assert w.wait_ready(), w.error
errors, sent = [], [0]
lock = threading.Lock()
stop = threading.Event()
def producer(k):
    rng = np.random.default_rng(k)
    while not stop.is_set():
        try:
            w.submit_ctrl(rng.uniform(-1, 1, w.sim.nu))
            with lock: sent[0] += 1
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        time.sleep(0.0005)
def consumer():
    last = -1
    while not stop.is_set():
        try:
            seq, s = w.latest_snapshot()
            if s is not None:
                assert seq >= last, 'snapshot seq went backwards'
                assert len(s.qpos) == len(s.joint_names)
                last = seq
            w.stats
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
threads = [threading.Thread(target=producer, args=(i,), daemon=True) for i in range(3)]
threads += [threading.Thread(target=consumer, daemon=True) for _ in range(3)]
sys.setswitchinterval(1e-6)
[t.start() for t in threads]
time.sleep(1.5)
stop.set(); [t.join() for t in threads]
sys.setswitchinterval(0.005)
st = w.stats
check('no exception across 6 ROS-side threads', not errors, f'errors={errors[:2]}')
check('physics kept ticking under load', st['ticks'] > 100, str(st))
# stats spans two locks, so the tick and cmd counters can be one tick apart.
check('commands coalesced, never replayed',
      st['cmd_used'] <= st['cmd_received']
      and abs(st['cmd_used'] + st['cmd_repeat'] + st['stale_ticks'] - st['ticks']) <= 2,
      f"sent={sent[0]} recv={st['cmd_received']} used={st['cmd_used']} "
      f"repeat={st['cmd_repeat']} stale={st['stale_ticks']} ticks={st['ticks']}")
check('robot moved (commands actually reached MjData)',
      abs(w.latest_snapshot()[1].qvel).max() > 1e-6)

# ---------------------------------------------------------------- test 3
# Reset requested from another thread lands between steps, not inside one.
home_q = make_sim().snapshot().qpos.copy()
rt = SimWorker(make_sim(), steps_per_cmd=N, use_viewer=False, realtime=True, cmd_timeout=0.0)
rt.start(); assert rt.wait_ready(), rt.error
time.sleep(0.2)
moved = rt.latest_snapshot()[1]
rt.request_reset()
deadline, hit = time.monotonic() + 2.0, False
while time.monotonic() < deadline and not hit:
    snap = rt.latest_snapshot()[1]
    hit = snap is not None and np.array_equal(snap.qpos, home_q) and snap.time == 0.0
rt.stop('test done'); rt.join()
check('reset from a ROS thread lands exactly on the home keyframe', hit,
      f'pre-reset t={moved.time:.3f}s')

# ---------------------------------------------------------------- test 4
# Command timeout: stop feeding, ctrl must fall back to zero.
before = w.stats['stale_ticks']
time.sleep(1.0)
after = w.stats
check('stale command falls back to zero ctrl', after['stale_ticks'] > before,
      f"stale {before} -> {after['stale_ticks']}")
check('ctrl actually zeroed', np.allclose(w.latest_snapshot()[1].ctrl, 0.0))

w.stop('test done'); w.join()
check('stop() ends the thread', w.finished and w.error is None)

# ---------------------------------------------------------------- test 5
# Realtime pacing: sim time must track wall time.
w = SimWorker(make_sim(), steps_per_cmd=N, use_viewer=False, realtime=True, cmd_timeout=0.0)
w.start(); assert w.wait_ready(), w.error
t0, s0 = time.monotonic(), w.latest_snapshot()[1].time
time.sleep(1.0)
wall, sim_dt = time.monotonic() - t0, w.latest_snapshot()[1].time - s0
w.stop('test done'); w.join()
check('realtime pacing keeps sim time within 5% of wall time',
      abs(sim_dt - wall) / wall < 0.05, f'wall={wall:.3f}s sim={sim_dt:.3f}s')

print('\n' + ('ALL PASS' if not fails else f'FAILURES: {fails}'))
sys.exit(1 if fails else 0)
