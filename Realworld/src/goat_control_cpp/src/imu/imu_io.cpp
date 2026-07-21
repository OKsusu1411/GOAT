#include "goat_control_cpp/imu/imu_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>

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

  msg.quat.w = q[0];
  msg.quat.x = q[1];
  msg.quat.y = q[2];
  msg.quat.z = q[3];

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

void ImuIO::open() {

}
void ImuIO::close(){

}
void ImuIO::read_loop(){

}