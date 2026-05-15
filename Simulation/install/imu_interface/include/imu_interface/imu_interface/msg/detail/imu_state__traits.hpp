// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from imu_interface:msg/ImuState.idl
// generated code does not contain a copyright notice

#ifndef IMU_INTERFACE__MSG__DETAIL__IMU_STATE__TRAITS_HPP_
#define IMU_INTERFACE__MSG__DETAIL__IMU_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "imu_interface/msg/detail/imu_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'quat'
#include "geometry_msgs/msg/detail/quaternion__traits.hpp"
// Member 'gyro'
// Member 'vel'
// Member 'mag'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace imu_interface
{

namespace msg
{

inline void to_flow_style_yaml(
  const ImuState & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: quat
  {
    out << "quat: ";
    to_flow_style_yaml(msg.quat, out);
    out << ", ";
  }

  // member: gyro
  {
    out << "gyro: ";
    to_flow_style_yaml(msg.gyro, out);
    out << ", ";
  }

  // member: vel
  {
    out << "vel: ";
    to_flow_style_yaml(msg.vel, out);
    out << ", ";
  }

  // member: mag
  {
    out << "mag: ";
    to_flow_style_yaml(msg.mag, out);
    out << ", ";
  }

  // member: time_ms
  {
    out << "time_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.time_ms, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ImuState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: quat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "quat:\n";
    to_block_style_yaml(msg.quat, out, indentation + 2);
  }

  // member: gyro
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gyro:\n";
    to_block_style_yaml(msg.gyro, out, indentation + 2);
  }

  // member: vel
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vel:\n";
    to_block_style_yaml(msg.vel, out, indentation + 2);
  }

  // member: mag
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mag:\n";
    to_block_style_yaml(msg.mag, out, indentation + 2);
  }

  // member: time_ms
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time_ms: ";
    rosidl_generator_traits::value_to_yaml(msg.time_ms, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ImuState & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace imu_interface

namespace rosidl_generator_traits
{

[[deprecated("use imu_interface::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const imu_interface::msg::ImuState & msg,
  std::ostream & out, size_t indentation = 0)
{
  imu_interface::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use imu_interface::msg::to_yaml() instead")]]
inline std::string to_yaml(const imu_interface::msg::ImuState & msg)
{
  return imu_interface::msg::to_yaml(msg);
}

template<>
inline const char * data_type<imu_interface::msg::ImuState>()
{
  return "imu_interface::msg::ImuState";
}

template<>
inline const char * name<imu_interface::msg::ImuState>()
{
  return "imu_interface/msg/ImuState";
}

template<>
struct has_fixed_size<imu_interface::msg::ImuState>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Quaternion>::value && has_fixed_size<geometry_msgs::msg::Vector3>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<imu_interface::msg::ImuState>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Quaternion>::value && has_bounded_size<geometry_msgs::msg::Vector3>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<imu_interface::msg::ImuState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // IMU_INTERFACE__MSG__DETAIL__IMU_STATE__TRAITS_HPP_
