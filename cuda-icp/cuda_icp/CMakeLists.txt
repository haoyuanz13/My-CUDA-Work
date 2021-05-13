#set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

# opencv
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# cuda
find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 --default-stream per-thread -Xcompiler -fopenmp ")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;--expt-relaxed-constexpr;-std=c++11")
list(APPEND icp_inc ${CUDA_INCLUDE_DIRS})
list(APPEND icp_lib ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# src
SET(icp_cuda_srcs  icp.cu scene/common.cu  scene/pcd_scene/pcd_scene.cu)
SET(icp_srcs  icp.cpp scene/common.cpp  scene/pcd_scene/pcd_scene.cpp)


CUDA_COMPILE(icp_cuda_objs ${icp_cuda_srcs})


# lib & test exe
add_library(cuda_icp #SHARED
    ${icp_srcs}
    ${icp_cuda_srcs}
    ${icp_cuda_objs} 
)
target_include_directories(cuda_icp PUBLIC ${icp_inc}  ${OpenCV_LIBS})
target_link_libraries(cuda_icp PUBLIC ${icp_lib} ${OpenCV_LIBS} )
