#include <stdio.h>
#include <math.h>

#include "kernel.h"


// To be used after calls that do not return an error code, ex. kernels to check kernel launch errors
void checkLastError(char const * func, const char *file, const int line, bool abort = true);
#define checkLastCUDAError(func) { checkLastError(func, __FILE__, __LINE__);  }
#define checkLastCUDAError_noAbort(func) { checkLastError(func, __FILE__, __LINE__, 0);  }
void checkLastError(char const * func, const char *file, const int line, bool abort) {
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        const char * errorMessage = cudaGetErrorString(code);
        fprintf(stderr, "CUDA error returned from \"%s\" at %s:%d, Error code: %d (%s)\n", func, file, line, code, errorMessage);
        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }
}


/*
 * normalize bgr image data and transfer the order to the bgr
 * 
 */
 __global__ void preprocess_norm_bgr2rgb_kernel(
                                const int batch_size,
                                const int data_row, 
                                const int data_col, 
                                const int data_channel, 
                                float *d_norm, 
                                const uint8_t* const d_in, 
                                const float* const d_mean, 
                                const float* const d_std_frac
                            ) {
    // 2D Index of current thread
    const int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    if(threadX >= data_col || threadY >= data_row * batch_size) {
        return;
    }

    // calculate batch id and yIndex wrt each single batch
    const int batch_id = static_cast<int>(threadY / data_row);
    const int batch_pixel_num = data_col * data_row * data_channel;
    const int yIndex = threadY % data_row;

    // calculate the d_in index wrt the d_in buffer
    // due to the order of d_in is the cv::Mat pointer, NHWC, the flattened data order is
    // [
    //     b0_0, g0_0, r0_0, ..., b0_N, g0_N, r0_N, 
    //     ...
    //     bM_0, gM_0, rM_0, ..., bM_N, gM_N, rM_N, 
    // ]
    // N is the pixel id of each batch, M is the batch id
    const int index_in = batch_id * batch_pixel_num + (threadX + yIndex * data_col) * data_channel;

    // normalize for all channels
    // in the output buffer, the data order is NCHW, 
    // [
    //    r0_0, ..., r0_N, g0_1, ..., g0_N, b0_1, ..., b0_N, 
    //    ...
    //    rM_0, ..., rM_N, gM_1, ..., gM_N, bM_1, ..., bM_N, 
    // ]
    for (int c = 0; c < data_channel; c ++) {
        int index_out = batch_id * batch_pixel_num + threadX + yIndex * data_col + c * data_col * data_row;
        d_norm[index_out] = static_cast<float>(d_in[index_in + data_channel - 1 - c]) / 255.0;  // bgr to rgb
        d_norm[index_out] -= d_mean[c];
        d_norm[index_out] *= d_std_frac[c];
    }
}


void normalize_bgr2rgb_cuda(const int batch_size, 
        const int data_row, const int data_col, 
        float *d_norm, uint8_t *d_in, 
        void *d_mean, void *d_std_frac, 
        cudaStream_t &stream) {

    // define block size
    dim3 block(16, 16);
    const int grid_x = (data_col - 1) / block.x + 1;
    const int grid_y = (data_row * batch_size - 1) / block.y + 1;
    const int channel = 3;
    dim3 grid(grid_x, grid_y);

    preprocess_norm_bgr2rgb_kernel<<<grid, block, 0, stream>>>(
        batch_size,
        data_row, data_col, channel, 
        static_cast<float*>(d_norm),
        static_cast<const uint8_t* const>(d_in), 
        static_cast<const float* const>(d_mean),
        static_cast<const float* const>(d_std_frac)
    );
}


/*
 * normalize gray image data
 * 
 */
__global__ void preprocess_norm_gray_kernel(
                                const int batch_size,
                                const int data_row, 
                                const int data_col, 
                                float *d_norm, 
                                const uint8_t *d_in, 
                                const float *d_mean,
                                const float *d_std_frac 
                            ) {
    // 2D Index of current thread
    const int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    if(threadX >= data_col || threadY >= data_row * batch_size) {
        return;
    }

    // calculate batch id and yIndex wrt each single batch
    const int batch_id = static_cast<int>(threadY / data_row);
    const int yIndex = threadY % data_row;

    // int index = threadX + threadY * data_col;
    const int index = batch_id * data_col * data_row + threadX + yIndex * data_col;
    
    // normalize
    d_norm[index] = static_cast<float>(d_in[index]) / 255.0;
    d_norm[index] -= *d_mean;
    d_norm[index] *= *d_std_frac;
}


void normalize_gray_cuda(const int batch_size, 
        const int data_row, const int data_col, 
        float *d_norm, uint8_t *d_in, 
        void *d_mean, void *d_std_frac, 
        cudaStream_t &stream) {

    // define threads size
    dim3 block(16, 16);
    const int grid_x = (data_col - 1) / block.x + 1;
    const int grid_y = (data_row * batch_size - 1) / block.y + 1;
    dim3 grid(grid_x, grid_y);

    preprocess_norm_gray_kernel<<<grid, block, 0, stream>>>(
        batch_size,
        data_row, data_col, 
        static_cast<float*>(d_norm),
        static_cast<const uint8_t* const>(d_in), 
        static_cast<const float* const>(d_mean),
        static_cast<const float* const>(d_std_frac)
    );
}


void _cudaDeviceSynchronize() {
    cudaDeviceSynchronize();
}


// end of this file