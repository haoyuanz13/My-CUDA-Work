#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>
#include <Eigen/Core>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#define cudaCheckError(e) {                                          \
  if(e != cudaSuccess) {                                              \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
    exit(0); \
  }                                                                 \
}


/// \brief keep the size and normalize data
void normalize_gray_cuda(const int batch_size, 
	const int data_row, const int data_col, 
	float *d_norm, uint8_t *d_in, 
	void *d_mean, void *d_std_frac,
	cudaStream_t &stream
);


/// \brief stop cpu working until the gpu work finish
void _cudaDeviceSynchronize();

// end of this file