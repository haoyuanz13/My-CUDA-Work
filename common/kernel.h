#pragma once

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#define cudaCheckError(e) {                                          \
  if(e != cudaSuccess) {                                              \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
    exit(0); \
  }                                                                 \
}


/// \brief resize op using bilinear interpolation, take the float type as the demo
void resize_cuda(const int batch_size, 
        const int input_row, const int input_col, 
        const int output_row, const int output_col, 
        const int channel_num, const bool align_corners, 
        float* d_out, float* d_in,
        cudaStream_t& stream);


/// \brief the padding kernel, no normalization and channels flip, but will do NHWC -> NCHW
/// \brief which is more suitable to the nn work
void center_aligned_padding_cuda(const int batch_size, 
        const int input_row, const int input_col, 
        const int output_row, const int output_col, 
        const int channel_num, const uint8_t padding_val, 
        uint8_t *d_out, uint8_t *d_in, 
        cudaStream_t &stream);


/// \brief keep the size and normalize bgr data, also transfer into the rgb order
void normalize_bgr2rgb_cuda(const int batch_size, 
        const int data_row, const int data_col, 
        float *d_norm, uint8_t *d_in, 
        void *d_mean, void *d_std_frac, 
        cudaStream_t &stream);


/// \brief keep the size and normalize data
void normalize_gray_cuda(const int batch_size, 
        const int data_row, const int data_col, 
        float *d_norm, uint8_t *d_in, 
        void *d_mean, void *d_std_frac,
        cudaStream_t &stream);


/// \brief stop cpu working until the gpu work finish
void _cudaDeviceSynchronize();

// end of this file
