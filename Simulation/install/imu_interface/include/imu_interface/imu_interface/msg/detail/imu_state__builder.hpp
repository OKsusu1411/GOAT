// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from imu_interface:msg/ImuState.idl
// generated code does not contain a copyright notice

#ifndef IMU_INTERFACE__MSG__DETAIL__IMU_STATE__BUILDER_HPP_
#define IMU_INTERFACE__MSG__DETAIL__IMU_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "imu_interface/msg/detail/imu_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace imu_interface
{

namespace msg
{

namespace builder
{

class Init_ImuState_time_ms
{
public:
  explicit Init_ImuState_time_ms(::imu_interface::msg::ImuState & msg)
  : msg_(msg)
  {}
  ::imu_interface::msg::ImuState time_ms(::imu_interface::msg::ImuState::_time_ms_type arg)
  {
    msg_.time_ms = std::move(arg);
    return std::move(msg_);
  }

private:
  ::imu_interface::msg::ImuState msg_;
};

class Init_ImuState_mag
{
public:
  explicit Init_ImuState_mag(::imu_interface::msg::ImuState & msg)
  : msg_(msg)
  {}
  Init_ImuState_time_ms mag(::imu_interface::msg::ImuState::_mag_type arg)
  {
    msg_.mag = std::move(arg);
    return Init_ImuState_time_ms(msg_);
  }

private:
  ::imu_interface::msg::ImuState msg_;
};

class Init_ImuState_vel
{
public:
  explicit Init_ImuState_vel(::imu_interface::msg::ImuState & msg)
  : msg_(msg)
  {}
  Init_ImuState_mag vel(::imu_interface::msg::ImuState::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_ImuState_mag(msg_);
  }

private:
  ::imu_interface::msg::ImuState msg_;
};

class Init_ImuState_gyro
{
public:
  explicit Init_ImuState_gyro(::imu_interface::msg::ImuState & msg)
  : msg_(msg)
  {}
  Init_ImuState_vel gyro(::imu_interface::msg::ImuState::_gyro_type arg)
  {
    msg_.gyro = std::move(arg);
    return Init_ImuState_vel(msg_);
  }

private:
  ::imu_interface::msg::ImuState msg_;
};

class Init_ImuState_quat
{
public:
  explicit Init_ImuState_quat(::imu_interface::msg::ImuState & msg)
  : msg_(msg)
  {}
  Init_ImuState_gyro quat(::imu_interface::msg::ImuState::_quat_type arg)
  {
    msg_.quat = std::move(arg);
    return Init_ImuState_gyro(msg_);
  }

private:
  ::imu_interface::msg::ImuState msg_;
};

class Init_ImuState_header
{
public:
  Init_ImuState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ImuState_quat header(::imu_interface::msg::ImuState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ImuState_quat(msg_);
  }

private:
  ::imu_interface::msg::ImuState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::imu_interface::msg::ImuState>()
{
  return imu_interface::msg::builder::Init_ImuState_header();
}

}  // namespace imu_interface

#endif  // IMU_INTERFACE__MSG__DETAIL__IMU_STATE__BUILDER_HPP_
