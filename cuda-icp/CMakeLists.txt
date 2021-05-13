cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 14)
project(icp_worker)
MESSAGE( STATUS "PROJECT_SRC_DIR = ${PROJECT_SOURCE_DIR}.")

#SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS "-fopenmp -fPIC -Wall -Wno-sign-compare $ENV{CXXFLAGS} ")  # add pic flag
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")


#--------------------------------------------------
# compile flags
#--------------------------------------------------
add_definitions(-DCUDA_ON)  # use cuda
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;--expt-relaxed-constexpr;-std=c++11")


#--------------------------------------------------
# dependence
#--------------------------------------------------
# cuda
find_package(CUDA REQUIRED)
list(APPEND CUDA_INCS ${CUDA_INCLUDE_DIRS})
list(APPEND CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})


# opencv
# find_package(OpenCV 3 REQUIRED)
# set(OpenCV_DIR /usr/local/opencv-3.4.5)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
MESSAGE( STATUS "opencv_include = ${OpenCV_INCLUDE_DIRS}.")
MESSAGE( STATUS "opencv_library = ${OpenCV_LIBS}.")

# pcl
# set(PCL_DIR /usr/local/pcl-1.9.1/share/pcl-1.9)
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
MESSAGE( STATUS "pcl_include = ${PCL_INCLUDE_DIRS}.")
MESSAGE( STATUS "pcl_library = ${PCL_LIBRARIES}.")

# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
MESSAGE( STATUS "eigen_include = ${EIGEN3_INCLUDE_DIR}.")


#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------
set(LIB_NAME_ICP_CUDA cuda_icp)
set(LIB_NAME_ICP_CPP cpp_icp)
set(LIB_NAME_ICP_WORKER icp_worker)
set(EXE_NAME_TEST demo_icp_worker)

# set srcs
set(CUDA_ICP_DIR ${PROJECT_SOURCE_DIR}/cuda_icp)
set(LIB_SRC_ICP_CU 
    ${PROJECT_SOURCE_DIR}/cuda_icp/icp.cu
    ${PROJECT_SOURCE_DIR}/cuda_icp/scene/common.cu
    ${PROJECT_SOURCE_DIR}/cuda_icp/scene/pcd_scene/pcd_scene.cu
)
set(LIB_SRC_ICP_CPP
    ${PROJECT_SOURCE_DIR}/cuda_icp/icp.cpp
    ${PROJECT_SOURCE_DIR}/cuda_icp/scene/common.cpp
    ${PROJECT_SOURCE_DIR}/cuda_icp/scene/pcd_scene/pcd_scene.cpp
)
set(LIB_SRC_ICP_WORKER 
    ${PROJECT_SOURCE_DIR}/src/helper.cpp
    ${PROJECT_SOURCE_DIR}/src/icp_worker.cpp
)
set(EXE_SRC_TEST ${PROJECT_SOURCE_DIR}/main/demo.cpp)

# include dirs
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCS}
)

# cuda icp lib, shared
cuda_add_library(${LIB_NAME_ICP_CUDA} SHARED
    ${LIB_SRC_ICP_CU}
)
target_link_libraries(${LIB_NAME_ICP_CUDA} 
    ${CUDA_LIBS} 
    ${OpenCV_LIBS}
)

# cpp icp lib, shared
add_library(${LIB_NAME_ICP_CPP} SHARED
    ${LIB_SRC_ICP_CPP}
)
target_link_libraries(${LIB_NAME_ICP_CPP} 
    ${CUDA_LIBS} 
    ${OpenCV_LIBS}
)

# icp worker lib, shared
add_library(${LIB_NAME_ICP_WORKER} SHARED
    ${LIB_SRC_ICP_WORKER}
)
target_include_directories(${LIB_NAME_ICP_WORKER} PUBLIC ${CUDA_ICP_DIR})
target_link_libraries(${LIB_NAME_ICP_WORKER}
    ${LIB_NAME_ICP_CUDA}
    ${LIB_NAME_ICP_CPP}
    ${PCL_LIBRARIES}
)

# test exe
add_executable(${EXE_NAME_TEST} ${EXE_SRC_TEST})
target_link_libraries(${EXE_NAME_TEST}
    ${LIB_NAME_ICP_WORKER}
)

