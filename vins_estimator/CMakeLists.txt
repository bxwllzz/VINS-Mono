cmake_minimum_required(VERSION 2.8.3)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
if("$ENV{ROS_DISTRO}" STREQUAL "kinetic")
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    message("cpp11")
elseif("$ENV{ROS_DISTRO}" STREQUAL "melodic")
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    message("cpp14")
endif()
#-DEIGEN_USE_MKL_ALL")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message(WARNING "Using ccache")
endif(CCACHE_FOUND)

find_package(catkin REQUIRED COMPONENTS
    rosbag
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    )

find_package(OpenCV REQUIRED)

message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_executable(vins_estimator
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/estimator_base.cpp
    src/feature_manager.cpp
    src/factor/base_odom_intergration.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/utility/ImuUtility.cpp)

target_link_libraries(vins_estimator ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(excalib
    src/excalib.cpp
    src/utility/utility.cpp
    src/odom-vio_ex_calib.cpp
)

target_link_libraries(excalib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(modify-bag
        src/modify_bag.cpp
        )
target_link_libraries(modify-bag ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(check-bag
        src/check_bag.cpp
        )
target_link_libraries(check-bag ${catkin_LIBRARIES} ${OpenCV_LIBS})
