# ImuStates.msg

# Header
std_msgs/Header header

# IMU data
geometry_msgs/Quaternion     quat
geometry_msgs/Vector3        gyro
geometry_msgs/Vector3        vel
geometry_msgs/Vector3        mag

float64                      time_ms
