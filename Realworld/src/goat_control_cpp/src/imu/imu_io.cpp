#include "goat_control_cpp/imu/imu_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <Eigen/Geometry>

// Constructor
ImuIO::ImuIO(ImuConfig cfg):cfg_(std::move(cfg)){
  latest_raw_vector_.assign(cfg_.expected_length, 0.0);
  open();  
}
ImuIO::ImuIO():ImuIO(ImuConfig{}) {}

// Destructor
ImuIO::~ImuIO() {close();}

// Read API
goat_api::msg::ImuState ImuIO::read_imu() {
  goat_api::msg::ImuState msg;

  // Raw data
  std::vector<double> data;
  bool have_data;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    have_data = has_valid_packet_;
    data = latest_raw_vector_;
  }
  if (!have_data) return msg;     // Exception catch

  // Apply IMU calibration
  const Eigen::Quaterniond quat_offset(cfg_.imu_offsets[0], cfg_.imu_offsets[1],
                                       cfg_.imu_offsets[2], cfg_.imu_offsets[3]);
  const Eigen::Quaterniond raw_quat(data[0], data[1], data[2], data[3]);
  const Eigen::Quaterniond q = quat_offset * raw_quat;

  msg.quat.w = q.w();
  msg.quat.x = q.x();
  msg.quat.y = q.y();
  msg.quat.z = q.z();

  // Degree to Radian
  constexpr double Deg2Rad = M_PI / 180.0;
  msg.gyro.x = data[4] * Deg2Rad;
  msg.gyro.y = data[5] * Deg2Rad;
  msg.gyro.z = data[6] * Deg2Rad;

  msg.vel.x = data[7];
  msg.vel.y = data[8];
  msg.vel.z = data[9];

  msg.mag.x = data[10];
  msg.mag.y = data[11];
  msg.mag.z = data[12];

  msg.time_ms = data[13];
  return msg;
}

// Open serial port + spawn reader thread. No-op if already open.
void ImuIO::open() {
  if (serial_port_ && serial_port_->isOpen()) return;

  serial_port_ = std::make_unique<serial::Serial>(
      cfg_.port,
      static_cast<uint32_t>(cfg_.baud_rate),
      serial::Timeout::simpleTimeout(static_cast<uint32_t>(cfg_.timeout_ms)));

  stop_flag_.store(false);
  reader_thread_ = std::thread(&ImuIO::read_loop, this);

  std::cout << "[IMU] opened serial: " << cfg_.port
            << " @ " << cfg_.baud_rate << " bps" << std::endl;
}

// Signal stop, join reader, close port. Idempotent.
void ImuIO::close(){
  stop_flag_.store(true);

  if (reader_thread_.joinable()) reader_thread_.join();

  if (serial_port_) {
    try {
      if (serial_port_->isOpen()) serial_port_->close();
    } catch (...) {
      // Match Python's swallow-on-close semantics
    }
    serial_port_.reset();
  }

  std::cout << "[IMU] closed" << std::endl;
}

// Dummy standalone test entry. Compiled only when IMU_IO_TEST_MAIN is defined
// (see CMakeLists.txt executable target `imu_io_test`). Opens the IMU on the
// port passed as argv[1] (default /dev/ttyUSB0) and prints each decoded frame.
#ifdef IMU_IO_TEST_MAIN
int main(int argc, char** argv) {
  ImuConfig cfg;
  if (argc >= 2) cfg.port = argv[1];
  cfg.imu_offsets = {1.0, 0.0, 0.0, 0.0};  // identity — no calibration applied

  ImuIO imu(std::move(cfg));

  // Give the reader a beat to pick up its first packet
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  while (true) {
    const auto m = imu.read_imu();
    std::cout << "quat=(" << m.quat.w << ", " << m.quat.x << ", "
              << m.quat.y << ", " << m.quat.z << ")  "
              << "gyro=(" << m.gyro.x << ", " << m.gyro.y << ", "
              << m.gyro.z << ") rad/s  "
              << "vel=(" << m.vel.x << ", " << m.vel.y << ", "
              << m.vel.z << ") m/s  "
              << "mag=(" << m.mag.x << ", " << m.mag.y << ", "
              << m.mag.z << ")  "
              << "t=" << m.time_ms << " ms"
              << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return 0;
}
#endif

// Background reader. Line format from IMU firmware:
//   *w,x,y,z,gx,gy,gz,vx,vy,vz,mx,my,mz,t_ms
// Each valid line replaces latest_raw_vector_ under mutex_.
void ImuIO::read_loop(){
  while (!stop_flag_.load()) {
    try {
      if (!serial_port_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }

      // readline blocks up to timeout_ms; strip whitespace/CR/LF
      std::string raw_line = serial_port_->readline(65536, "\n");
      const auto first = raw_line.find_first_not_of(" \t\r\n");
      if (first == std::string::npos) continue;
      const auto last = raw_line.find_last_not_of(" \t\r\n");
      raw_line = raw_line.substr(first, last - first + 1);

      if (raw_line.front() != cfg_.start_char) continue;

      // Drop start char, split on ',', parse each field to double
      const std::string payload = raw_line.substr(1);
      std::vector<double> float_values;
      float_values.reserve(cfg_.expected_length);

      std::stringstream ss(payload);
      std::string field;
      bool parse_ok = true;
      while (std::getline(ss, field, ',')) {
        try {
          float_values.push_back(std::stod(field));
        } catch (const std::exception&) {
          parse_ok = false;
          break;
        }
      }
      if (!parse_ok) continue;
      if (float_values.size() != cfg_.expected_length) continue;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_raw_vector_ = std::move(float_values);
        has_valid_packet_ = true;
      }
    } catch (const std::exception& e) {
      std::cerr << "[IMU] read error: " << e.what() << std::endl;
      std::this_thread::sleep_for(
          std::chrono::duration<float>(cfg_.read_sleep_sec_on_error));
    }
  }
}