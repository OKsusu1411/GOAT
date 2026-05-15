// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from imu_interface:msg/ImuState.idl
// generated code does not contain a copyright notice
#include "imu_interface/msg/detail/imu_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `quat`
#include "geometry_msgs/msg/detail/quaternion__functions.h"
// Member `gyro`
// Member `vel`
// Member `mag`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
imu_interface__msg__ImuState__init(imu_interface__msg__ImuState * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    imu_interface__msg__ImuState__fini(msg);
    return false;
  }
  // quat
  if (!geometry_msgs__msg__Quaternion__init(&msg->quat)) {
    imu_interface__msg__ImuState__fini(msg);
    return false;
  }
  // gyro
  if (!geometry_msgs__msg__Vector3__init(&msg->gyro)) {
    imu_interface__msg__ImuState__fini(msg);
    return false;
  }
  // vel
  if (!geometry_msgs__msg__Vector3__init(&msg->vel)) {
    imu_interface__msg__ImuState__fini(msg);
    return false;
  }
  // mag
  if (!geometry_msgs__msg__Vector3__init(&msg->mag)) {
    imu_interface__msg__ImuState__fini(msg);
    return false;
  }
  // time_ms
  return true;
}

void
imu_interface__msg__ImuState__fini(imu_interface__msg__ImuState * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // quat
  geometry_msgs__msg__Quaternion__fini(&msg->quat);
  // gyro
  geometry_msgs__msg__Vector3__fini(&msg->gyro);
  // vel
  geometry_msgs__msg__Vector3__fini(&msg->vel);
  // mag
  geometry_msgs__msg__Vector3__fini(&msg->mag);
  // time_ms
}

bool
imu_interface__msg__ImuState__are_equal(const imu_interface__msg__ImuState * lhs, const imu_interface__msg__ImuState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // quat
  if (!geometry_msgs__msg__Quaternion__are_equal(
      &(lhs->quat), &(rhs->quat)))
  {
    return false;
  }
  // gyro
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->gyro), &(rhs->gyro)))
  {
    return false;
  }
  // vel
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->vel), &(rhs->vel)))
  {
    return false;
  }
  // mag
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->mag), &(rhs->mag)))
  {
    return false;
  }
  // time_ms
  if (lhs->time_ms != rhs->time_ms) {
    return false;
  }
  return true;
}

bool
imu_interface__msg__ImuState__copy(
  const imu_interface__msg__ImuState * input,
  imu_interface__msg__ImuState * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // quat
  if (!geometry_msgs__msg__Quaternion__copy(
      &(input->quat), &(output->quat)))
  {
    return false;
  }
  // gyro
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->gyro), &(output->gyro)))
  {
    return false;
  }
  // vel
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->vel), &(output->vel)))
  {
    return false;
  }
  // mag
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->mag), &(output->mag)))
  {
    return false;
  }
  // time_ms
  output->time_ms = input->time_ms;
  return true;
}

imu_interface__msg__ImuState *
imu_interface__msg__ImuState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  imu_interface__msg__ImuState * msg = (imu_interface__msg__ImuState *)allocator.allocate(sizeof(imu_interface__msg__ImuState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(imu_interface__msg__ImuState));
  bool success = imu_interface__msg__ImuState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
imu_interface__msg__ImuState__destroy(imu_interface__msg__ImuState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    imu_interface__msg__ImuState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
imu_interface__msg__ImuState__Sequence__init(imu_interface__msg__ImuState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  imu_interface__msg__ImuState * data = NULL;

  if (size) {
    data = (imu_interface__msg__ImuState *)allocator.zero_allocate(size, sizeof(imu_interface__msg__ImuState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = imu_interface__msg__ImuState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        imu_interface__msg__ImuState__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
imu_interface__msg__ImuState__Sequence__fini(imu_interface__msg__ImuState__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      imu_interface__msg__ImuState__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

imu_interface__msg__ImuState__Sequence *
imu_interface__msg__ImuState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  imu_interface__msg__ImuState__Sequence * array = (imu_interface__msg__ImuState__Sequence *)allocator.allocate(sizeof(imu_interface__msg__ImuState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = imu_interface__msg__ImuState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
imu_interface__msg__ImuState__Sequence__destroy(imu_interface__msg__ImuState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    imu_interface__msg__ImuState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
imu_interface__msg__ImuState__Sequence__are_equal(const imu_interface__msg__ImuState__Sequence * lhs, const imu_interface__msg__ImuState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!imu_interface__msg__ImuState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
imu_interface__msg__ImuState__Sequence__copy(
  const imu_interface__msg__ImuState__Sequence * input,
  imu_interface__msg__ImuState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(imu_interface__msg__ImuState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    imu_interface__msg__ImuState * data =
      (imu_interface__msg__ImuState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!imu_interface__msg__ImuState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          imu_interface__msg__ImuState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!imu_interface__msg__ImuState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
