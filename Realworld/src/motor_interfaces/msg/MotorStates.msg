# MotorStates.msg

# Header
std_msgs/Header header

# Motor data
float32[] temperature_c
float32[] iq_amp
float32[] speed_dps
int32[]   encoder_raw
int32[]   single_turn_raw
int64[]   multi_turn_raw
uint8[]   error_flags
uint8[]   motor_state