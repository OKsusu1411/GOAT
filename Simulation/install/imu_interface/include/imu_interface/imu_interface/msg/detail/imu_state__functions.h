// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from imu_interface:msg/ImuState.idl
// generated code does not contain a copyright notice

#ifndef IMU_INTERFACE__MSG__DETAIL__IMU_STATE__FUNCTIONS_H_
#define IMU_INTERFACE__MSG__DETAIL__IMU_STATE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "imu_interface/msg/rosidl_generator_c__visibility_control.h"

#include "imu_interface/msg/detail/imu_state__struct.h"

/// Initialize msg/ImuState message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * imu_interface__msg__ImuState
 * )) before or use
 * imu_interface__msg__ImuState__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
bool
imu_interface__msg__ImuState__init(imu_interface__msg__ImuState * msg);

/// Finalize msg/ImuState message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
void
imu_interface__msg__ImuState__fini(imu_interface__msg__ImuState * msg);

/// Create msg/ImuState message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * imu_interface__msg__ImuState__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
imu_interface__msg__ImuState *
imu_interface__msg__ImuState__create();

/// Destroy msg/ImuState message.
/**
 * It calls
 * imu_interface__msg__ImuState__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
void
imu_interface__msg__ImuState__destroy(imu_interface__msg__ImuState * msg);

/// Check for msg/ImuState message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
bool
imu_interface__msg__ImuState__are_equal(const imu_interface__msg__ImuState * lhs, const imu_interface__msg__ImuState * rhs);

/// Copy a msg/ImuState message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
bool
imu_interface__msg__ImuState__copy(
  const imu_interface__msg__ImuState * input,
  imu_interface__msg__ImuState * output);

/// Initialize array of msg/ImuState messages.
/**
 * It allocates the memory for the number of elements and calls
 * imu_interface__msg__ImuState__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
bool
imu_interface__msg__ImuState__Sequence__init(imu_interface__msg__ImuState__Sequence * array, size_t size);

/// Finalize array of msg/ImuState messages.
/**
 * It calls
 * imu_interface__msg__ImuState__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
void
imu_interface__msg__ImuState__Sequence__fini(imu_interface__msg__ImuState__Sequence * array);

/// Create array of msg/ImuState messages.
/**
 * It allocates the memory for the array and calls
 * imu_interface__msg__ImuState__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
imu_interface__msg__ImuState__Sequence *
imu_interface__msg__ImuState__Sequence__create(size_t size);

/// Destroy array of msg/ImuState messages.
/**
 * It calls
 * imu_interface__msg__ImuState__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
void
imu_interface__msg__ImuState__Sequence__destroy(imu_interface__msg__ImuState__Sequence * array);

/// Check for msg/ImuState message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
bool
imu_interface__msg__ImuState__Sequence__are_equal(const imu_interface__msg__ImuState__Sequence * lhs, const imu_interface__msg__ImuState__Sequence * rhs);

/// Copy an array of msg/ImuState messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_imu_interface
bool
imu_interface__msg__ImuState__Sequence__copy(
  const imu_interface__msg__ImuState__Sequence * input,
  imu_interface__msg__ImuState__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // IMU_INTERFACE__MSG__DETAIL__IMU_STATE__FUNCTIONS_H_
