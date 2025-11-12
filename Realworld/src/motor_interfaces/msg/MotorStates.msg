# MotorStates.msg
# 이 메시지는 모든 모터의 상태 정보를 포함합니다.

# 메시지가 생성된 시간과 좌표계 정보
std_msgs/Header header

# 각 필드는 모터 개수만큼의 길이를 갖는 배열(리스트)입니다.
float32[] temperature_c
float32[] iq_amp
float32[] speed_dps
int32[]   encoder_raw
int32[]   single_turn_raw
int64[]   multi_turn_raw
uint8[]   error_flags
uint8[]   motor_state