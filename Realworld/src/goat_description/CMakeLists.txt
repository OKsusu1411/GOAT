cmake_minimum_required(VERSION 3.8)
project(goat_description)

find_package(ament_cmake REQUIRED)

# launch, urdf, meshes 폴더를 install/share/goat_description/ 밑으로 복사
install(
  DIRECTORY launch urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
