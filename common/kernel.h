#pragma once

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#define cudaCheckError(e) {                                          \
  if(e != cudaSuccess) {                                              \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
    exit(0); \
  }                                                                 \
}


/// \brief matrix transpose using the shared memory and naive implementation
void mat_transpose_shared_mem_naive_cuda(const int row, const int col,
        float *d_mat, float *d_out,
        cudaStream_t &stream);


/// \brief matrix transpose using the global memory
void mat_transpose_cuda(const int row, const int col,
        float *d_mat, float *d_out,
        cudaStream_t &stream);


/// \brief matrix multiplication using the shared memory
void mat_multiply_share_mem_cuda(const int row_A, const int col_A, const int col_B,  
        float *d_mat_A, float *d_mat_B, float *d_mat_C, 
        cudaStream_t &stream);


/// \brief matrix multiplication using the global memory
void mat_multiply_cuda(const int row_A, const int col_A, const int col_B,  
        float *d_mat_A, float *d_mat_B, float *d_mat_C, 
        cudaStream_t &stream);


/// \brief reduce method to extract feature map, using shared memeory
void max_ele_extract_reduce_shared_mem_cuda(const int batch_size, 
        const int feat_map_row, const int feat_map_col, 
        float *d_feat_map, 
        int *d_state_ids, float *d_max_vals, float *d_exp_val_sum, 
        cudaStream_t &stream);


/// \brief implement the cv::copyTo and speed up
void copyto_hwc2chw_cuda(const int data_dst_row, const int data_dst_col, 
        const int data_roi_row, const int data_roi_col, 
        const int data_channel, 
        const int offset_row, const int offset_col, 
        uint8_t *d_out, uint8_t *d_src, uint8_t *d_roi, 
        cudaStream_t &stream);


/// \brief resize op using bilinear interpolation, take the float type as the demo
void resize_cuda(const int batch_size, 
        const int input_row, const int input_col, 
        const int output_row, const int output_col, 
        const int channel_num, const bool align_corners, 
        float *d_out, float *d_in,
        cudaStream_t &stream);


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
