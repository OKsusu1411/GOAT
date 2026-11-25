#ifndef MOTOR_IO_H_
#define MOTOR_IO_H_

#include <vector>

#include "sensor_msgs/msg/joint_state.hpp"

class MotorIO {
  public:
  explicit MotorIO();
  ~MotorIO();
  // Copy constructor
  MotorIO(const MotorIO&) = delete;
  MotorIO& operator=(const MotorIO&) = delete;

  // Main api
  void can_open();
  void write_motor(std::vector<double>& torque_cmd_nm);
  sensor_msgs::msg::JointState read_motor();
  void can_close();
};

#endif // MOTOR_IO_H_