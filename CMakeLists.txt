# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
cmake_minimum_required(VERSION 3.5)
project(my_cuda LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD  11)   # c++11
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
set(USE_OPENCV ON)

# Enable compiler warnings
if (CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -ggdb -O0 -Wno-deprecated-declarations -Wno-unused-function ")
else(CMAKE_BUILD_TYPE "Release")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-deprecated-declarations -Wno-unused-function")
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------
set(EXE_NAME_GRAY_NORM ${PROJECT_NAME}_gray_norm)
set(EXE_NAME_BGR2RGB_NORM ${PROJECT_NAME}_bgr2rgb_norm)
set(CU_LIB_NAME cu_kernels)

# set exe files
set(EXE_SOURCES_GRAY_NORM
  ${PROJECT_SOURCE_DIR}/src/main_gray_normalize.cpp
)
set(EXE_SOURCES_BGR2RGB_NORM
  ${PROJECT_SOURCE_DIR}/src/main_bgr2rgb_normalize.cpp
)
set(CU_SOURCES
  ${PROJECT_SOURCE_DIR}/common/kernel.cu
)

# set exe includes
set(EXE_INCLUDE
  ${PROJECT_SOURCE_DIR}/common
)


# CUDA and cuDNN deps
find_package(CUDA 10.0 REQUIRED)
MESSAGE(STATUS "FOUND CUDNN Lib: ${CUDNN_LIBRARY}")

# OpenCV deps
if(USE_OPENCV)
find_package(OpenCV 3.3 REQUIRED)
if(OpenCV_FOUND)
  MESSAGE(STATUS "Found OpenCV:${OpenCV_INCLUDE_DIRS}, libs:${OpenCV_LIBS}")
  MESSAGE(STATUS "Found OpenCV Version: ${OpenCV_VERSION}")
endif()
endif(USE_OPENCV)


# --------------------------------
# Converter executable
# --------------------------------
include_directories(
  ${CUDA_INCLUDE_DIRS}
)
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda -std=c++11")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR})
CUDA_ADD_LIBRARY(${CU_LIB_NAME} SHARED ${CU_SOURCES})


# executable 
add_executable(${EXE_NAME_GRAY_NORM} ${EXE_SOURCES_GRAY_NORM})
target_include_directories(${EXE_NAME_GRAY_NORM} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_GRAY_NORM}
  ${CUDA_LIBRARIES}
  ${CUDNN_LIBRARY}
  ${CU_LIB_NAME}
)

add_executable(${EXE_NAME_BGR2RGB_NORM} ${EXE_SOURCES_BGR2RGB_NORM})
target_include_directories(${EXE_NAME_BGR2RGB_NORM} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_BGR2RGB_NORM}
  ${CUDA_LIBRARIES}
  ${CUDNN_LIBRARY}
  ${CU_LIB_NAME}
)

if(USE_OPENCV)
  target_include_directories(${EXE_NAME_GRAY_NORM} PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${EXE_NAME_GRAY_NORM} ${OpenCV_LIBS})
  target_include_directories(${EXE_NAME_BGR2RGB_NORM} PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${EXE_NAME_BGR2RGB_NORM} ${OpenCV_LIBS})
endif()
