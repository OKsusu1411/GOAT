#ifndef MOTOR_IO_H_
#define MOTOR_IO_H_

#include <array>

#include "sensor_msgs/msg/joint_state.hpp"

class MotorIO {
  public:
  // Constructor
  explicit MotorIO();

  // Destructor
  ~MotorIO();

  // Main api
  void write_motor(int& torque_cmd_nm);
  sensor_msgs::msg::JointState read_motor();
  void can_close();
};

#endif // MOTOR_IO_H_