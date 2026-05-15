// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from imu_interface:msg/ImuState.idl
// generated code does not contain a copyright notice

#ifndef IMU_INTERFACE__MSG__DETAIL__IMU_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define IMU_INTERFACE__MSG__DETAIL__IMU_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "imu_interface/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "imu_interface/msg/detail/imu_state__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace imu_interface
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_imu_interface
cdr_serialize(
  const imu_interface::msg::ImuState & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_imu_interface
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  imu_interface::msg::ImuState & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_imu_interface
get_serialized_size(
  const imu_interface::msg::ImuState & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_imu_interface
max_serialized_size_ImuState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace imu_interface

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_imu_interface
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, imu_interface, msg, ImuState)();

#ifdef __cplusplus
}
#endif

#endif  // IMU_INTERFACE__MSG__DETAIL__IMU_STATE__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
