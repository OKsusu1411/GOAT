// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from imu_interface:msg/ImuState.idl
// generated code does not contain a copyright notice

#ifndef IMU_INTERFACE__MSG__DETAIL__IMU_STATE__STRUCT_H_
#define IMU_INTERFACE__MSG__DETAIL__IMU_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'quat'
#include "geometry_msgs/msg/detail/quaternion__struct.h"
// Member 'gyro'
// Member 'vel'
// Member 'mag'
#include "geometry_msgs/msg/detail/vector3__struct.h"

/// Struct defined in msg/ImuState in the package imu_interface.
/**
  * ImuState.msg
 */
typedef struct imu_interface__msg__ImuState
{
  /// Header
  std_msgs__msg__Header header;
  /// IMU data
  geometry_msgs__msg__Quaternion quat;
  geometry_msgs__msg__Vector3 gyro;
  geometry_msgs__msg__Vector3 vel;
  geometry_msgs__msg__Vector3 mag;
  double time_ms;
} imu_interface__msg__ImuState;

// Struct for a sequence of imu_interface__msg__ImuState.
typedef struct imu_interface__msg__ImuState__Sequence
{
  imu_interface__msg__ImuState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} imu_interface__msg__ImuState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // IMU_INTERFACE__MSG__DETAIL__IMU_STATE__STRUCT_H_
