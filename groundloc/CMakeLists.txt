cmake_minimum_required(VERSION 3.22)
project(groundloc)

######################################
## Find catkin macros and libraries ##
######################################

set (CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

find_package(OpenCV REQUIRED)

# Do not warn for unused returned values
add_compile_options(-Wno-unused-result)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

set(DEPS
  rclpy
  rclcpp
  rclcpp_components
  geometry_msgs
  std_msgs
  sensor_msgs      
  cv_bridge
  image_transport
  visualization_msgs
  tf2
  tf2_ros
  tf2_msgs
  tf2_sensor_msgs
)


include_directories(include)

#############
## Install ##
#############

install(PROGRAMS
    scripts/collect_geotiff.py
    DESTINATION lib/${PROJECT_NAME}/
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY res
    DESTINATION share/${PROJECT_NAME}/
)
ament_package()
