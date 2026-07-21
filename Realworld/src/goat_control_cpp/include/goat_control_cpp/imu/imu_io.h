#ifndef IMU_IO_H_
#define IMU_IO_H_

#include <chrono>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <array>

#include "goat_control_cpp/serial/serial.h"
#include "goat_api/msg/imu_state.hpp"

// IMU configuration
struct ImuConfig {
  std::string port = "/dev/ttyUSB0";
  std::array<double, 4> imu_offsets;
  uint8_t expected_length = 14;
  int baud_rate = 115200;
  int timeout_ms = 1000;
  char start_char = '*';
  float read_sleep_sec_on_error = 0.01;
};

class ImuIO{
 public:
  // Constructor
  explicit ImuIO(ImuConfig cfg);
  ImuIO();
  
  // Copy constructor
  ImuIO(const ImuIO&)            = delete;
  ImuIO& operator=(const ImuIO&) = delete;

  // Destructor
  ~ImuIO();

  // Read API
  goat_api::msg::ImuState read_imu();

 private:
  ImuConfig cfg_;
  std::thread reader_thread_;
  std::unique_ptr<serial::Serial> serial_port_;
  std::atomic<bool> stop_flag_{false};

  std::mutex mutex_;
  std::vector<double> latest_raw_vector_;
  bool has_valid_packet_ = false;

  void open();
  void close();
  void read_loop();
};

#endif // IMU_IO_H_