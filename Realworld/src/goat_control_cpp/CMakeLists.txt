cmake_minimum_required(VERSION 3.8)
project(goat_control_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies -----------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(goat_api REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# --- serial (vendored port of wjwwood/serial, Unix backend only) -----------
add_library(serial
  src/serial/serial.cc
  src/serial/impl/unix.cc
  src/serial/impl/list_ports/list_ports_linux.cc
)
target_include_directories(serial PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(serial PUBLIC Threads::Threads)

# --- imu_io library --------------------------------------------------------
add_library(imu_io
  src/imu/imu_io.cpp
)
target_include_directories(imu_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(imu_io PUBLIC
  serial
  Eigen3::Eigen
  Threads::Threads
)
ament_target_dependencies(imu_io PUBLIC
  rclcpp
  goat_api
)

# --- imu_io_test executable (dummy standalone tester) ---------------------
# Re-compiles imu_io.cpp with IMU_IO_TEST_MAIN defined so its guarded main()
# becomes the entry point. Run with: ros2 run goat_control_cpp imu_io_test [port]
add_executable(imu_io_test src/imu/imu_io.cpp)
target_compile_definitions(imu_io_test PRIVATE IMU_IO_TEST_MAIN)
target_include_directories(imu_io_test PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(imu_io_test
  serial
  Eigen3::Eigen
  Threads::Threads
)
ament_target_dependencies(imu_io_test
  rclcpp
  goat_api
)

# --- Install ---------------------------------------------------------------
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS serial imu_io
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  TARGETS imu_io_test
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp goat_api Eigen3)
ament_export_include_directories(include)

# --- Testing ---------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
