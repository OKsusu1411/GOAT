# ImuState.msg

# Header
std_msgs/Header header

# IMU data
geometry_msgs/Quaternion     quat       # Quaternion - unitless
geometry_msgs/Vector3        gyro       # Gyroscope(Angular velocity) - rad/s
geometry_msgs/Vector3        vel        # Linear velocity - m/s
geometry_msgs/Vector3        mag        # Magnetometer(not used) - ?

float64                      time_ms    # Time - ms
