// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from imu_interface:msg/ImuState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "imu_interface/msg/detail/imu_state__rosidl_typesupport_introspection_c.h"
#include "imu_interface/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "imu_interface/msg/detail/imu_state__functions.h"
#include "imu_interface/msg/detail/imu_state__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `quat`
#include "geometry_msgs/msg/quaternion.h"
// Member `quat`
#include "geometry_msgs/msg/detail/quaternion__rosidl_typesupport_introspection_c.h"
// Member `gyro`
// Member `vel`
// Member `mag`
#include "geometry_msgs/msg/vector3.h"
// Member `gyro`
// Member `vel`
// Member `mag`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  imu_interface__msg__ImuState__init(message_memory);
}

void imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_fini_function(void * message_memory)
{
  imu_interface__msg__ImuState__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_member_array[6] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(imu_interface__msg__ImuState, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "quat",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(imu_interface__msg__ImuState, quat),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "gyro",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(imu_interface__msg__ImuState, gyro),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vel",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(imu_interface__msg__ImuState, vel),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mag",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(imu_interface__msg__ImuState, mag),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "time_ms",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(imu_interface__msg__ImuState, time_ms),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_members = {
  "imu_interface__msg",  // message namespace
  "ImuState",  // message name
  6,  // number of fields
  sizeof(imu_interface__msg__ImuState),
  imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_member_array,  // message members
  imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_init_function,  // function to initialize message memory (memory has to be allocated)
  imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_type_support_handle = {
  0,
  &imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_imu_interface
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, imu_interface, msg, ImuState)() {
  imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Quaternion)();
  imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  if (!imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_type_support_handle.typesupport_identifier) {
    imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &imu_interface__msg__ImuState__rosidl_typesupport_introspection_c__ImuState_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
