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
// std::array<uint8_t, 4> pack_int32_le_signed (int64_t value);
// std::array<uint8_t, 4> pack_uint32_le       (int64_t value);

// Physical <-> LSB helpers (use current scales; call set_mg_unit_scales first).
int32_t current_amp_to_lsb          (double current_amp);
// int32_t angle_deg_to_lsb            (double angle_deg);
// int32_t speed_deg_per_sec_to_lsb    (double speed_deg_per_sec);

// double  lsb_to_current_amp          (int32_t current_lsb);
// double  lsb_to_angle_deg            (int32_t angle_lsb);
// double  lsb_to_speed_deg_per_sec    (int32_t speed_lsb);

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

#endif // CAN_H_