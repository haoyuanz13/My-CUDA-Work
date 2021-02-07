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
set(EXE_NAME_PADDING ${PROJECT_NAME}_center_aligned_padding)
set(EXE_NAME_COPYTO ${PROJECT_NAME}_copyto)
set(EXE_NAME_RESIZE ${PROJECT_NAME}_bilinear_inter_resize)
set(EXE_NAME_POST_CLS ${PROJECT_NAME}_reduce_post_cls)
set(EXE_NAME_MAT_MULTI ${PROJECT_NAME}_matrix_multi)
set(EXE_NAME_MAT_TRANS ${PROJECT_NAME}_matrix_trans)
set(CU_LIB_NAME cu_kernels)

# set exe files
set(EXE_SOURCES_GRAY_NORM ${PROJECT_SOURCE_DIR}/src/main_gray_normalize.cpp)
set(EXE_SOURCES_BGR2RGB_NORM ${PROJECT_SOURCE_DIR}/src/main_bgr2rgb_normalize.cpp)
set(EXE_SOURCES_PADDING ${PROJECT_SOURCE_DIR}/src/main_center_aligned_padding.cpp)
set(EXE_SOURCES_COPYTO ${PROJECT_SOURCE_DIR}/src/main_copyto.cpp)
set(EXE_SOURCES_RESIZE ${PROJECT_SOURCE_DIR}/src/main_bilinear_inter_resize.cpp)
set(EXE_SOURCES_POST_CLS ${PROJECT_SOURCE_DIR}/src/main_reduce_postprocess_cls.cpp)
set(EXE_SOURCES_MAT_MULTI ${PROJECT_SOURCE_DIR}/src/main_matrix_multiplication.cpp)
set(EXE_SOURCES_MAT_TRANS ${PROJECT_SOURCE_DIR}/src/main_matrix_transpose.cpp)
set(CU_SOURCES ${PROJECT_SOURCE_DIR}/common/kernel.cu)

# set exe includes
set(EXE_INCLUDE
  ${PROJECT_SOURCE_DIR}/common
)

# OpenCV deps
if(USE_OPENCV)
find_package(OpenCV 3.3 REQUIRED)
if(OpenCV_FOUND)
  MESSAGE(STATUS "Found OpenCV:${OpenCV_INCLUDE_DIRS}, libs:${OpenCV_LIBS}")
  MESSAGE(STATUS "Found OpenCV Version: ${OpenCV_VERSION}")
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_libraries(${OpenCV_LIBS})
endif()
endif(USE_OPENCV)

# CUDA and cuDNN deps
find_package(CUDA 10.0 REQUIRED)
if(CUDA_FOUND)
  MESSAGE(STATUS "FOUND CUDNN Lib: ${CUDNN_LIBRARY}")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda -std=c++11")
  CUDA_ADD_LIBRARY(${CU_LIB_NAME} SHARED ${CU_SOURCES})
  TARGET_INCLUDE_DIRECTORIES(${CU_LIB_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(${CU_LIB_NAME}
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
  )
endif()

# Eigen
find_package(Eigen3 REQUIRED)


# --------------------------------
# Converter executable
# --------------------------------
# executable 
add_executable(${EXE_NAME_GRAY_NORM} ${EXE_SOURCES_GRAY_NORM})
target_include_directories(${EXE_NAME_GRAY_NORM} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_GRAY_NORM}
  ${CU_LIB_NAME}
  ${PROJECT_SOURCE_DIR}/common/utils.cpp
)

add_executable(${EXE_NAME_BGR2RGB_NORM} ${EXE_SOURCES_BGR2RGB_NORM})
target_include_directories(${EXE_NAME_BGR2RGB_NORM} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_BGR2RGB_NORM}
  ${CU_LIB_NAME}
  ${PROJECT_SOURCE_DIR}/common/utils.cpp
)

add_executable(${EXE_NAME_PADDING} ${EXE_SOURCES_PADDING})
target_include_directories(${EXE_NAME_PADDING} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_PADDING}
  ${CU_LIB_NAME}
  ${PROJECT_SOURCE_DIR}/common/utils.cpp
)

add_executable(${EXE_NAME_COPYTO} ${EXE_SOURCES_COPYTO})
target_include_directories(${EXE_NAME_COPYTO} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_COPYTO}
  ${CU_LIB_NAME}
  ${PROJECT_SOURCE_DIR}/common/utils.cpp
)

add_executable(${EXE_NAME_RESIZE} ${EXE_SOURCES_RESIZE})
target_include_directories(${EXE_NAME_RESIZE} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_RESIZE}
  ${CU_LIB_NAME}
  ${PROJECT_SOURCE_DIR}/common/utils.cpp
)

add_executable(${EXE_NAME_POST_CLS} ${EXE_SOURCES_POST_CLS})
target_include_directories(${EXE_NAME_POST_CLS} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_POST_CLS}
  ${CU_LIB_NAME}
  ${PROJECT_SOURCE_DIR}/common/utils.cpp
)

add_executable(${EXE_NAME_MAT_MULTI} ${EXE_SOURCES_MAT_MULTI})
target_include_directories(${EXE_NAME_MAT_MULTI} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_MAT_MULTI}
  ${CU_LIB_NAME}
  ${PROJECT_SOURCE_DIR}/common/utils.cpp
)

add_executable(${EXE_NAME_MAT_TRANS} ${EXE_SOURCES_MAT_TRANS})
target_include_directories(${EXE_NAME_MAT_TRANS} PUBLIC ${EXE_INCLUDE})
target_link_libraries(${EXE_NAME_MAT_TRANS}
  ${CU_LIB_NAME}
  ${PROJECT_SOURCE_DIR}/common/utils.cpp
)