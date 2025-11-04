cmake_minimum_required(VERSION 3.22)
set(CUDACXX /usr/local/cuda-12.8/bin/nvcc) 
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc) 
project(quatro)

set(CMAKE_CXX_STANDARD 17)
set(Torch_DIR /home/ubuntu/.local/libtorch/share/cmake/Torch)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()

# disabled due to false positives in combination with assert statements
add_compile_options(-Wno-unused-but-set-variable)

# bugs in quatro (clang)
add_compile_options(-Wno-reorder-ctor)
# gcc
add_compile_options(-Wno-reorder)
add_compile_options(-Wno-maybe-uninitialized)
add_compile_options(-Wno-type-limits)
# dynamic size arrays in clang (might be different for gcc)
add_compile_options(-Wno-vla-cxx-extension)

find_package(rclcpp)
find_package(rclcpp_components REQUIRED)
find_package(rclpy)
find_package(std_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros 2.4.1 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

include(3rdparty/find_dependencies.cmake)

find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenMP)
find_package(PCL REQUIRED)
find_package(Boost 1.54 REQUIRED)
find_package(Eigen3 3.2 QUIET REQUIRED NO_MODULE)
find_package(GDAL REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_library(NVJPEG_LIBRARY nvjpeg ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPIG_LIBRARY nppig ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPIDEI_LIBRARY nppidei ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_library(NVINFER NAMES nvinfer)
find_library(NVINFERPLUGIN NAMES nvinfer_plugin)
find_library(NVPARSERS NAMES nvparsers)


set(DEPS
  rclpy
  rclcpp
  rclcpp_components
  std_msgs
  image_transport
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
  nav_msgs
  tf2
  tf2_ros
  tf2_msgs
  tf2_sensor_msgs
)

rosidl_generate_interfaces(${PROJECT_NAME}
        "msg/CloudInfo.msg"
         DEPENDENCIES std_msgs
 )

include_directories(
        include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
link_directories(${CMAKE_CUDA_INCLUDE_DIRS})


#file(DOWNLOAD https://urserver.kaist.ac.kr/publicdata/quatro/000540.bin ${CMAKE_CURRENT_LIST_DIR}/materials/000540.bin)
#file(DOWNLOAD https://urserver.kaist.ac.kr/publicdata/quatro/001319.bin ${CMAKE_CURRENT_LIST_DIR}/materials/001319.bin)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()



####### Executable ####### 
set(teaser_src
        src/graph.cc
        src/teaser_utils/feature_matcher.cc
        src/teaser_utils/fpfh.cc
        )

add_executable(quatro_node_cuda nodes/quatro_node_cuda.cpp ${teaser_src})
add_executable(quatro_node_sift nodes/quatro_node_sift.cpp ${teaser_src})
#add_executable(run_example nodes/run_global_registration.cpp ${teaser_src})

ament_target_dependencies(quatro_node_cuda
        ${DEPS}
        PCL
        OpenMP
        OpenCV
)
target_link_libraries(quatro_node_cuda
        pmc::pmc
         nvinfer nvinfer_plugin
        ${NVJPEG_LIBRARY} ${NPPIG_LIBRARY} ${NPPC_LIBRARY} ${NPPIDEI_LIBRARY} ${CULIBOS} "${TORCH_LIBRARIES}"
)
ament_target_dependencies(quatro_node_sift
        ${DEPS}
        PCL
        OpenMP
        OpenCV
)
target_link_libraries(quatro_node_sift
        pmc::pmc
)


#target_link_libraries(run_example
#        PUBLIC
#        ${PCL_LIBRARY_DIRS}
#        ${catkin_LIBRARIES}
#        stdc++fs
#        pmc::pmc
#        )

#############
## Install ##
#############

install(TARGETS
  quatro_node_cuda
	ARCHIVE DESTINATION lib/${PROJECT_NAME}
	LIBRARY DESTINATION lib/${PROJECT_NAME}
	RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS
  quatro_node_sift
	ARCHIVE DESTINATION lib/${PROJECT_NAME}
	LIBRARY DESTINATION lib/${PROJECT_NAME}
	RUNTIME DESTINATION lib/${PROJECT_NAME}
)
  
install(
	DIRECTORY launch
	DESTINATION share/${PROJECT_NAME}/
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
