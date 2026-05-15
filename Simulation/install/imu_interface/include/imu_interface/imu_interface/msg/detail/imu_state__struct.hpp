// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from imu_interface:msg/ImuState.idl
// generated code does not contain a copyright notice

#ifndef IMU_INTERFACE__MSG__DETAIL__IMU_STATE__STRUCT_HPP_
#define IMU_INTERFACE__MSG__DETAIL__IMU_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'quat'
#include "geometry_msgs/msg/detail/quaternion__struct.hpp"
// Member 'gyro'
// Member 'vel'
// Member 'mag'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__imu_interface__msg__ImuState __attribute__((deprecated))
#else
# define DEPRECATED__imu_interface__msg__ImuState __declspec(deprecated)
#endif

namespace imu_interface
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ImuState_
{
  using Type = ImuState_<ContainerAllocator>;

  explicit ImuState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    quat(_init),
    gyro(_init),
    vel(_init),
    mag(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time_ms = 0.0;
    }
  }

  explicit ImuState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    quat(_alloc, _init),
    gyro(_alloc, _init),
    vel(_alloc, _init),
    mag(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time_ms = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _quat_type =
    geometry_msgs::msg::Quaternion_<ContainerAllocator>;
  _quat_type quat;
  using _gyro_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _gyro_type gyro;
  using _vel_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _vel_type vel;
  using _mag_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _mag_type mag;
  using _time_ms_type =
    double;
  _time_ms_type time_ms;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__quat(
    const geometry_msgs::msg::Quaternion_<ContainerAllocator> & _arg)
  {
    this->quat = _arg;
    return *this;
  }
  Type & set__gyro(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->gyro = _arg;
    return *this;
  }
  Type & set__vel(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->vel = _arg;
    return *this;
  }
  Type & set__mag(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->mag = _arg;
    return *this;
  }
  Type & set__time_ms(
    const double & _arg)
  {
    this->time_ms = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    imu_interface::msg::ImuState_<ContainerAllocator> *;
  using ConstRawPtr =
    const imu_interface::msg::ImuState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<imu_interface::msg::ImuState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<imu_interface::msg::ImuState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      imu_interface::msg::ImuState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<imu_interface::msg::ImuState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      imu_interface::msg::ImuState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<imu_interface::msg::ImuState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<imu_interface::msg::ImuState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<imu_interface::msg::ImuState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__imu_interface__msg__ImuState
    std::shared_ptr<imu_interface::msg::ImuState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__imu_interface__msg__ImuState
    std::shared_ptr<imu_interface::msg::ImuState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ImuState_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->quat != other.quat) {
      return false;
    }
    if (this->gyro != other.gyro) {
      return false;
    }
    if (this->vel != other.vel) {
      return false;
    }
    if (this->mag != other.mag) {
      return false;
    }
    if (this->time_ms != other.time_ms) {
      return false;
    }
    return true;
  }
  bool operator!=(const ImuState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ImuState_

// alias to use template instance with default allocator
using ImuState =
  imu_interface::msg::ImuState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace imu_interface

#endif  // IMU_INTERFACE__MSG__DETAIL__IMU_STATE__STRUCT_HPP_
