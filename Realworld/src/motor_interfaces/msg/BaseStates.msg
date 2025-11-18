# BaseStates.msg
# 이 메시지는 모든 모터의 상태 정보를 포함합니다.

# 메시지가 생성된 시간과 좌표계 정보
std_msgs/Header header

# 각 필드는 모터 개수만큼의 길이를 갖는 배열(리스트)입니다.
geometry_msgs/Quaternion     quat
geometry_msgs/Vector3        gyro
geometry_msgs/Vector3        acc
geometry_msgs/Vector3        mag

float64                      time_ms
