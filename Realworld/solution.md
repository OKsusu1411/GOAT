# Step 1 — `can.py` + `protocol.py` → `motor/can.{h,cpp}`

## Conclusion

Merge `can.py` (transport, 318 LOC) and `protocol.py` (encoding, ~150 live LOC) into **one header + one source file** under the C++ package:

- `src/goat_control_cpp/include/goat_control_cpp/motor/can.h` — replaces the empty stub.
- `src/goat_control_cpp/src/motor/can.cpp` — new.

Everything lives in namespace `goat::motor`. No third-party CAN library — direct Linux SocketCAN (`<linux/can.h>` + POSIX socket) as agreed. Kernel-header includes stay in the `.cpp`; the header stays clean (`Frame` is a POCO wrapper so nothing above the transport layer needs to know about `struct can_frame`).

Deprecated payload builders (0xA0/A2/A3/A4/A5/A6/A7/A8, etc.) are omitted.

## Logic

Contract preserved 1:1 from Python — I explained this in the previous walkthrough, so just the delta:

| Python | C++ |
|---|---|
| `python-can` `Bus` | Raw SocketCAN socket, `int fd_` |
| `bus.send(Message(id, data))` | `::write(fd_, &frame, sizeof(frame))` |
| `bus.recv(timeout=...)` | `::poll(fd, timeout_ms)` + `::read(fd_, &frame, sizeof(frame))` |
| `threading.Event` | `KeyEvent` = `{ mutex; condition_variable; bool set; }`, held by `std::shared_ptr` so aliasing is two map entries pointing at the same event |
| `threading.Lock` | `std::mutex` |
| `dict[(int,int), …]` | `std::unordered_map<uint32_t, …>` with key = `(arb_id << 8) \| cmd_byte` (11-bit id + 8-bit cmd fits comfortably) |
| MGUnitScales singleton | Namespace-scope `static` inside `can.cpp`, `set_mg_unit_scales`/`get_mg_unit_scales` |
| `int.to_bytes(..., "little")` | `htole16` / `htole32` from `<endian.h>` |

Two things worth calling out that aren't obvious from the Python:

1. **Stale-frame drain in `blocking_txrx`** — we do it with a nonblocking `::recv(fd, MSG_DONTWAIT)` loop instead of a Python-`recv(timeout=0)` loop. Same effect.
2. **Reader-thread poll cadence** — Python uses `bus.recv(timeout=0.05)` so `stop_event` is checked every 50 ms. C++ mirrors that with `::poll(..., 50)`.

## Code

### 1. `src/goat_control_cpp/include/goat_control_cpp/motor/can.h`

```cpp
#ifndef CAN_H_
#define CAN_H_

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace goat::motor {

// ============================================================================
// Section 1 — Protocol constants and encoders (was protocol.py)
// ============================================================================

// MG-series CAN IDs for one motor node.
struct CanIds {
  uint32_t tx_id;   // host → motor: 0x140 + node_id
  uint32_t rx_id;   // motor → host: 0x180 + node_id
};

// Return CanIds for a given node id (1..8 in this project).
CanIds mg_ids(int node_id);

// Empty 7-byte payload (used by all "read" commands: 0x9A, 0x9C, 0x92, 0x94).
inline constexpr std::array<uint8_t, 7> E7{};

// Unit conversion scales between LSB (wire) and physical units.
// Loaded once at startup from YAML via set_mg_unit_scales(); read via get.
struct MGUnitScales {
  double motor_current_amp_per_lsb;    // A / LSB
  double angle_deg_per_lsb;            // deg / LSB
  double speed_deg_per_sec_per_lsb;    // (deg/s) / LSB
};

void         set_mg_unit_scales(const MGUnitScales& scales);
MGUnitScales get_mg_unit_scales();

// Little-endian packers. Return fixed-size byte arrays (no allocation).
// The uint variants saturate on overflow to mirror Python's clamp+pack.
std::array<uint8_t, 2> pack_int16_le_signed (int32_t value);
std::array<uint8_t, 2> pack_uint16_le       (int32_t value);
std::array<uint8_t, 4> pack_int32_le_signed (int64_t value);
std::array<uint8_t, 4> pack_uint32_le       (int64_t value);

// Physical <-> LSB helpers (use current scales; call set_mg_unit_scales first).
int32_t current_amp_to_lsb          (double current_amp);
int32_t angle_deg_to_lsb            (double angle_deg);
int32_t speed_deg_per_sec_to_lsb    (double speed_deg_per_sec);

double  lsb_to_current_amp          (int32_t current_lsb);
double  lsb_to_angle_deg            (int32_t angle_lsb);
double  lsb_to_speed_deg_per_sec    (int32_t speed_lsb);

// Torque-mode payload builder for command 0xA1.
//   - Clamps to ±(4096 × amp_per_lsb) then to ±4096 LSB (double clamp).
//   - Returns just the 2-byte iq field, LE, signed int16.
std::array<uint8_t, 2> pack_iq_from_amp(double current_amp);

// Build the full 7-byte payload for 0xA1 command:
//   [00, 00, 00, iq_lo, iq_hi, 00, 00]
// Prepend 0xA1 in front of this to form the 8-byte CAN data field.
std::array<uint8_t, 7> payload_torque_mode_from_amp(double current_amp);

// ============================================================================
// Section 2 — Transport layer (was can.py)
// ============================================================================

// POCO CAN frame — decouples callers from <linux/can.h>.
struct Frame {
  uint32_t             arb_id{0};   // 11-bit standard id
  uint8_t              dlc{0};      // 1..8
  std::array<uint8_t, 8> data{};
};

// Arrival event for one reply key. mutex/cv guard `set`. Two map entries can
// point at the same KeyEvent (via shared_ptr) — the alias trick that fixes
// the "motor replies on tx_id" case.
struct KeyEvent {
  std::mutex              mutex;
  std::condition_variable cv;
  bool                    set{false};

  void clear() {
    std::lock_guard<std::mutex> lock(mutex);
    set = false;
  }
  // Blocks until `set` becomes true or `deadline` elapses. Idempotent-safe.
  bool wait_until(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_until(lock, deadline, [this]{ return set; });
  }
  void notify() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      set = true;
    }
    cv.notify_all();
  }
};

// One instance per physical CAN bus (can0, can1).
// Lifecycle:
//   open() → blocking_txrx() during init → start_reader_thread() → hot path
//   uses send_only() + event_for_key()+get_latest_frame(). stop_reader_thread()
//   + close() on shutdown.
class CanInterface {
 public:
  explicit CanInterface(std::string channel);
  ~CanInterface();

  CanInterface(const CanInterface&)            = delete;
  CanInterface& operator=(const CanInterface&) = delete;

  // Open the raw CAN socket bound to `channel_`. Throws on failure.
  void open();
  // Stop reader (if running), close socket. Idempotent.
  void close();

  // Reader-thread mode toggle. Must run AFTER all init blocking_txrx() calls.
  void start_reader_thread();
  void stop_reader_thread();
  bool is_reader_running() const;

  // Hot-path fire-and-forget send. Logs on error, does not throw.
  void send_only(uint32_t arb_id, const uint8_t* data, uint8_t dlc);

  // Init-time synchronous request/response. Drains stale frames, sends,
  // waits up to `timeout` for a matching frame. Returns std::nullopt on
  // timeout. Must not be called while the reader thread is running.
  std::optional<Frame> blocking_txrx(
      uint32_t tx_id,
      uint32_t rx_id,
      uint8_t  cmd_byte,
      const std::array<uint8_t, 7>& payload7,
      std::chrono::milliseconds timeout,
      bool accept_rx_id       = true,
      bool accept_tx_echo_diff = true);

  // Look up the last cached frame for (arb_id, cmd_byte). std::nullopt if
  // nothing has arrived on that key yet.
  std::optional<Frame> get_latest_frame(uint32_t arb_id, uint8_t cmd_byte);

  // Lazily allocate + return the arrival event for one reply key.
  std::shared_ptr<KeyEvent> event_for_key(uint32_t arb_id, uint8_t cmd_byte);

  // Point two reply keys at ONE shared KeyEvent. Handles the "motor replies
  // on tx_id instead of rx_id" case: reader sets whichever key arrives, waiter
  // on either key wakes. Call at setup, before the hot path arms events.
  std::shared_ptr<KeyEvent> alias_event_keys(
      uint32_t arb_id_a, uint8_t cmd_a,
      uint32_t arb_id_b, uint8_t cmd_b);

 private:
  // Packed (arb_id << 8) | cmd_byte — fits any 11-bit id + 8-bit cmd in 32 bits.
  static uint32_t make_key(uint32_t arb_id, uint8_t cmd_byte) {
    return (arb_id << 8) | cmd_byte;
  }

  // Reader thread body: poll → read → cache → notify event.
  void read_loop();

  // Drain any pending frames in the kernel RX queue (nonblocking). Used by
  // blocking_txrx to purge the previous motor's stale reply before sending.
  int drain_rx(int max_frames = 200);

  // Diagnostic: log the first N unique (arb_id, cmd_byte) keys we see.
  // Reveals whether motors reply on rx_id or tx_id on this hardware.
  void maybe_log_first_key(uint32_t arb_id, uint8_t cmd_byte);

  std::string channel_;
  int         fd_{-1};

  std::thread          reader_thread_;
  std::atomic<bool>    stop_flag_{false};

  std::mutex           cache_mutex_;
  std::unordered_map<uint32_t, Frame> latest_frames_;
  uint64_t             rx_frame_count_{0};
  std::unordered_set<uint32_t> first_keys_seen_;

  std::mutex           events_mutex_;
  std::unordered_map<uint32_t, std::shared_ptr<KeyEvent>> events_;

  // Serializes blocking_txrx() callers so A's response can't get consumed
  // by B's recv() during the init phase.
  std::mutex           txrx_mutex_;
};

}  // namespace goat::motor

#endif  // CAN_H_
```

### 2. `src/goat_control_cpp/src/motor/can.cpp`

```cpp
#include "goat_control_cpp/motor/can.h"

#include <endian.h>
#include <errno.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace goat::motor {

// ============================================================================
// Section 1 — Protocol (was protocol.py)
// ============================================================================

namespace {

// Live singleton. Defaults match protocol.py's _DEFAULT_MG_UNIT_SCALES.
MGUnitScales g_scales{
    .motor_current_amp_per_lsb = 66.0 / 4096.0,
    .angle_deg_per_lsb         = 0.001,
    .speed_deg_per_sec_per_lsb = 0.01,
};

std::mutex g_scales_mutex;

}  // namespace

CanIds mg_ids(int node_id) {
  return CanIds{
      .tx_id = static_cast<uint32_t>(0x140 + node_id),
      .rx_id = static_cast<uint32_t>(0x180 + node_id),
  };
}

void set_mg_unit_scales(const MGUnitScales& scales) {
  std::lock_guard<std::mutex> lock(g_scales_mutex);
  g_scales = scales;
}

MGUnitScales get_mg_unit_scales() {
  std::lock_guard<std::mutex> lock(g_scales_mutex);
  return g_scales;
}

std::array<uint8_t, 2> pack_int16_le_signed(int32_t value) {
  const uint16_t u = static_cast<uint16_t>(static_cast<int16_t>(value));
  const uint16_t le = htole16(u);
  std::array<uint8_t, 2> out{};
  std::memcpy(out.data(), &le, 2);
  return out;
}

std::array<uint8_t, 2> pack_uint16_le(int32_t value) {
  if (value < 0)          value = 0;
  if (value > 0xFFFF)     value = 0xFFFF;
  const uint16_t le = htole16(static_cast<uint16_t>(value));
  std::array<uint8_t, 2> out{};
  std::memcpy(out.data(), &le, 2);
  return out;
}

std::array<uint8_t, 4> pack_int32_le_signed(int64_t value) {
  const uint32_t u  = static_cast<uint32_t>(static_cast<int32_t>(value));
  const uint32_t le = htole32(u);
  std::array<uint8_t, 4> out{};
  std::memcpy(out.data(), &le, 4);
  return out;
}

std::array<uint8_t, 4> pack_uint32_le(int64_t value) {
  if (value < 0)               value = 0;
  if (value > 0xFFFFFFFFLL)    value = 0xFFFFFFFFLL;
  const uint32_t le = htole32(static_cast<uint32_t>(value));
  std::array<uint8_t, 4> out{};
  std::memcpy(out.data(), &le, 4);
  return out;
}

// Physical <-> LSB. `round` (not truncate) — matches Python's int(round(...)).
int32_t current_amp_to_lsb(double current_amp) {
  return static_cast<int32_t>(
      std::llround(current_amp / get_mg_unit_scales().motor_current_amp_per_lsb));
}
int32_t angle_deg_to_lsb(double angle_deg) {
  return static_cast<int32_t>(
      std::llround(angle_deg / get_mg_unit_scales().angle_deg_per_lsb));
}
int32_t speed_deg_per_sec_to_lsb(double speed_deg_per_sec) {
  return static_cast<int32_t>(
      std::llround(speed_deg_per_sec / get_mg_unit_scales().speed_deg_per_sec_per_lsb));
}

double lsb_to_current_amp(int32_t current_lsb) {
  return current_lsb * get_mg_unit_scales().motor_current_amp_per_lsb;
}
double lsb_to_angle_deg(int32_t angle_lsb) {
  return angle_lsb * get_mg_unit_scales().angle_deg_per_lsb;
}
double lsb_to_speed_deg_per_sec(int32_t speed_lsb) {
  return speed_lsb * get_mg_unit_scales().speed_deg_per_sec_per_lsb;
}

// Double clamp: (1) physical amp range, (2) LSB int16 range post-round.
std::array<uint8_t, 2> pack_iq_from_amp(double current_amp) {
  constexpr int32_t max_iq_lsb = 4096;
  const double max_iq_amp = max_iq_lsb * get_mg_unit_scales().motor_current_amp_per_lsb;

  double clamped_amp = std::max(std::min(current_amp, max_iq_amp), -max_iq_amp);
  int32_t iq_lsb     = current_amp_to_lsb(clamped_amp);
  iq_lsb = std::max(std::min(iq_lsb, max_iq_lsb), -max_iq_lsb);

  return pack_int16_le_signed(iq_lsb);
}

std::array<uint8_t, 7> payload_torque_mode_from_amp(double current_amp) {
  const auto iq_bytes = pack_iq_from_amp(current_amp);
  return {0x00, 0x00, 0x00, iq_bytes[0], iq_bytes[1], 0x00, 0x00};
}

// ============================================================================
// Section 2 — Transport (was can.py)
// ============================================================================

CanInterface::CanInterface(std::string channel)
    : channel_(std::move(channel)) {}

CanInterface::~CanInterface() { close(); }

void CanInterface::open() {
  if (fd_ >= 0) return;

  fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (fd_ < 0) {
    throw std::runtime_error(
        "[CAN] socket() failed on " + channel_ + ": " + std::strerror(errno));
  }

  struct ifreq ifr{};
  std::strncpy(ifr.ifr_name, channel_.c_str(), IFNAMSIZ - 1);
  if (::ioctl(fd_, SIOCGIFINDEX, &ifr) < 0) {
    const int err = errno;
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
        "[CAN] ioctl(SIOCGIFINDEX) failed on " + channel_ + ": " + std::strerror(err));
  }

  struct sockaddr_can addr{};
  addr.can_family  = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;
  if (::bind(fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    const int err = errno;
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
        "[CAN] bind() failed on " + channel_ + ": " + std::strerror(err));
  }

  std::cout << "[CAN] opened: socketcan:" << channel_ << std::endl;
}

void CanInterface::close() {
  stop_reader_thread();

  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
    std::cout << "[CAN] closed: " << channel_ << std::endl;
  }
}

void CanInterface::start_reader_thread() {
  if (reader_thread_.joinable()) return;
  if (fd_ < 0) {
    throw std::runtime_error("[CAN] start_reader_thread() before open() on " + channel_);
  }
  stop_flag_.store(false);
  reader_thread_ = std::thread(&CanInterface::read_loop, this);
  std::cout << "[CAN] reader thread started on " << channel_ << std::endl;
}

void CanInterface::stop_reader_thread() {
  stop_flag_.store(true);
  if (reader_thread_.joinable()) reader_thread_.join();
}

bool CanInterface::is_reader_running() const {
  return reader_thread_.joinable() && !stop_flag_.load();
}

void CanInterface::send_only(uint32_t arb_id, const uint8_t* data, uint8_t dlc) {
  if (fd_ < 0) {
    std::cerr << "[CAN] send_only on closed bus " << channel_ << std::endl;
    return;
  }
  struct can_frame f{};
  f.can_id  = arb_id & CAN_SFF_MASK;   // 11-bit standard id
  f.can_dlc = std::min<uint8_t>(dlc, 8);
  std::memcpy(f.data, data, f.can_dlc);

  const ssize_t n = ::write(fd_, &f, sizeof(f));
  if (n != static_cast<ssize_t>(sizeof(f))) {
    std::cerr << "[CAN] send_only failed on " << channel_
              << " (id=0x" << std::hex << arb_id << std::dec
              << "): " << std::strerror(errno) << std::endl;
  }
}

std::optional<Frame> CanInterface::blocking_txrx(
    uint32_t tx_id, uint32_t rx_id, uint8_t cmd_byte,
    const std::array<uint8_t, 7>& payload7,
    std::chrono::milliseconds timeout,
    bool accept_rx_id, bool accept_tx_echo_diff) {

  if (fd_ < 0) return std::nullopt;

  std::lock_guard<std::mutex> lock(txrx_mutex_);

  // (1) Purge stale RX frames left by the previous motor.
  const int drained = drain_rx();
  if (drained > 0) {
    std::cout << "[CAN] " << channel_ << " drained " << drained
              << " stale RX frames before blocking_txrx" << std::endl;
  }

  // (2) Compose + send.
  struct can_frame tx_frame{};
  tx_frame.can_id  = tx_id & CAN_SFF_MASK;
  tx_frame.can_dlc = 8;
  tx_frame.data[0] = cmd_byte;
  std::memcpy(&tx_frame.data[1], payload7.data(), 7);
  if (::write(fd_, &tx_frame, sizeof(tx_frame)) != static_cast<ssize_t>(sizeof(tx_frame))) {
    std::cerr << "[CAN] blocking_txrx send failed on " << channel_ << ": "
              << std::strerror(errno) << std::endl;
    return std::nullopt;
  }

  // (3) Poll → read → filter, until deadline.
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return std::nullopt;

    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now).count();
    struct pollfd pfd{fd_, POLLIN, 0};
    const int pr = ::poll(&pfd, 1, static_cast<int>(std::min<long>(remaining, 50)));
    if (pr <= 0) continue;                     // timeout tick or interrupted
    if (!(pfd.revents & POLLIN)) continue;

    struct can_frame rx{};
    const ssize_t n = ::read(fd_, &rx, sizeof(rx));
    if (n != static_cast<ssize_t>(sizeof(rx))) continue;
    if (rx.can_dlc != 8) continue;
    if (rx.data[0] != cmd_byte) continue;

    const uint32_t rid = rx.can_id & CAN_SFF_MASK;
    const bool is_rx_id = (rid == (rx_id & CAN_SFF_MASK));
    const bool is_tx_id = (rid == (tx_id & CAN_SFF_MASK));

    // Pure loopback echo (same tx_id + identical data) is always rejected.
    const bool is_pure_echo =
        is_tx_id && (std::memcmp(rx.data, tx_frame.data, 8) == 0);
    if (is_pure_echo) continue;

    if (accept_rx_id && is_rx_id) {
      Frame out{rid, rx.can_dlc, {}};
      std::memcpy(out.data.data(), rx.data, 8);
      return out;
    }
    if (accept_tx_echo_diff && is_tx_id) {
      Frame out{rid, rx.can_dlc, {}};
      std::memcpy(out.data.data(), rx.data, 8);
      return out;
    }
    // Frame from unrelated key — drop, keep waiting.
  }
}

int CanInterface::drain_rx(int max_frames) {
  int drained = 0;
  struct can_frame f{};
  while (drained < max_frames) {
    const ssize_t n = ::recv(fd_, &f, sizeof(f), MSG_DONTWAIT);
    if (n <= 0) break;
    ++drained;
  }
  return drained;
}

std::optional<Frame> CanInterface::get_latest_frame(uint32_t arb_id, uint8_t cmd_byte) {
  const uint32_t key = make_key(arb_id, cmd_byte);
  std::lock_guard<std::mutex> lock(cache_mutex_);
  auto it = latest_frames_.find(key);
  if (it == latest_frames_.end()) return std::nullopt;
  return it->second;
}

std::shared_ptr<KeyEvent> CanInterface::event_for_key(uint32_t arb_id, uint8_t cmd_byte) {
  const uint32_t key = make_key(arb_id, cmd_byte);
  std::lock_guard<std::mutex> lock(events_mutex_);
  auto it = events_.find(key);
  if (it != events_.end()) return it->second;
  auto ev = std::make_shared<KeyEvent>();
  events_.emplace(key, ev);
  return ev;
}

std::shared_ptr<KeyEvent> CanInterface::alias_event_keys(
    uint32_t arb_id_a, uint8_t cmd_a,
    uint32_t arb_id_b, uint8_t cmd_b) {
  const uint32_t ka = make_key(arb_id_a, cmd_a);
  const uint32_t kb = make_key(arb_id_b, cmd_b);
  std::lock_guard<std::mutex> lock(events_mutex_);

  auto find_existing = [&](uint32_t k) -> std::shared_ptr<KeyEvent> {
    auto it = events_.find(k);
    return it == events_.end() ? nullptr : it->second;
  };
  auto ev = find_existing(ka);
  if (!ev) ev = find_existing(kb);
  if (!ev) ev = std::make_shared<KeyEvent>();

  events_[ka] = ev;
  events_[kb] = ev;
  return ev;
}

void CanInterface::read_loop() {
  while (!stop_flag_.load()) {
    struct pollfd pfd{fd_, POLLIN, 0};
    const int pr = ::poll(&pfd, 1, 50);       // 50 ms — matches Python cadence
    if (pr <= 0) continue;
    if (!(pfd.revents & POLLIN)) continue;

    struct can_frame rx{};
    const ssize_t n = ::read(fd_, &rx, sizeof(rx));
    if (n != static_cast<ssize_t>(sizeof(rx))) continue;
    if (rx.can_dlc == 0) continue;

    const uint32_t arb = rx.can_id & CAN_SFF_MASK;
    const uint8_t  cmd = rx.data[0];
    const uint32_t key = make_key(arb, cmd);

    Frame frame{arb, rx.can_dlc, {}};
    std::memcpy(frame.data.data(), rx.data, 8);

    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      latest_frames_[key] = frame;
      ++rx_frame_count_;
    }

    // Wake anyone waiting on this exact key.
    std::shared_ptr<KeyEvent> ev;
    {
      std::lock_guard<std::mutex> lock(events_mutex_);
      auto it = events_.find(key);
      if (it != events_.end()) ev = it->second;
    }
    if (ev) ev->notify();

    maybe_log_first_key(arb, cmd);
  }
}

void CanInterface::maybe_log_first_key(uint32_t arb_id, uint8_t cmd_byte) {
  std::lock_guard<std::mutex> lock(cache_mutex_);   // reuses cache_mutex_ for the set
  if (first_keys_seen_.size() >= 16) return;
  const uint32_t key = make_key(arb_id, cmd_byte);
  if (!first_keys_seen_.insert(key).second) return;

  char buf[128];
  std::snprintf(buf, sizeof(buf),
                "[CAN] %s first frame for arb_id=0x%03X cmd=0x%02X (total=%llu)",
                channel_.c_str(), arb_id, cmd_byte,
                static_cast<unsigned long long>(rx_frame_count_));
  std::cout << buf << std::endl;
}

}  // namespace goat::motor
```

### 3. `CMakeLists.txt` addition (append inside the existing `imu_io` library target — do NOT touch anything else)

```cmake
# Add the motor source to the same library (single-lib decision, per the plan).
target_sources(imu_io PRIVATE src/motor/can.cpp)
```

*(Note: this reuses the existing `imu_io` library target. Once Step 5 lands we'll rename the target to `goat_control_cpp` to match its actual scope. For now this keeps CMake surgical — one `target_sources` line, no renames, no new targets.)*

## Notes on things I intentionally did NOT do

- **`MotorParams`, `MotorDriver`** — those are Step 2/3 (motor_driver.py port). Not in this file.
- **Deprecated payload builders** — 0xA0 / A2 / A3 / A4 / A5 / A6 / A7 / A8 skipped, as agreed.
- **Bitrate configuration** — the Python `bitrate=` arg to `can.Bus` is a no-op on Linux SocketCAN (the interface is brought up externally via `setup_can_test.sh` with `ip link set canX up type can bitrate 1000000`). C++ inherits the same assumption; nothing to configure at runtime.
- **CAN-FD** — not needed. Sticking with `can_frame` (16 bytes) and 11-bit standard IDs.
- **Logger injection** — using `std::cout`/`std::cerr` for now to match `imu_io.cpp` style. Swap for `rclcpp::Logger` when we wire this to a real ROS node.
- **No unit tests** — Step 2 will introduce a small `can_loopback_test` executable against `vcan0` as a smoke check.

## Ready to apply
Say go and I'll apply the header, the .cpp, and the one-line CMake addition. Nothing else touched.
