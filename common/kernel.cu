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

template <typename T>
__device__ __forceinline__ static T area_pixel_compute_source_index(T scale,
                                        int dst_index,
                                        bool align_corners,
                                        bool cubic = false) {
    if (align_corners) {
        return scale * dst_index;
    }
    else {  // cube align, geometry center align
        T src_idx = scale * (dst_index + static_cast<T>(0.5)) - static_cast<T>(0.5);
        return (!cubic && src_idx < static_cast<T>(0)) ? static_cast<T>(0): src_idx;
    }
}


template <typename T>
__host__ __forceinline__ T area_pixel_compute_scale(int input_size, int output_size, bool align_corners) {
    if(output_size > 1) {
        return align_corners ? static_cast<T>(input_size - 1) / (output_size - 1) : static_cast<T>(input_size) / output_size;
    }
    else {
        return static_cast<T>(0);
    }
}


/*
 * max elements extraction for the classification scenario
 * using shared mem to implement the reduce algorithm
 */
__global__ void max_ele_extract_reduce_shared_mem_kernel(
                                const int batch_size, 
                                const int feat_map_row, 
                                const int feat_map_col, 
                                float *d_feat_map, 
                                int *d_state_ids, 
                                float *d_max_vals, 
                                float *d_exp_val_sum
                            ) {

    // obtain block and thread index
    const int state_idx = blockIdx.x;
    const int prob_idx = threadIdx.x;
    if (state_idx >= batch_size * feat_map_row || prob_idx >= blockDim.x) {
        return;
    }

    // calculate feat map index
    const int feat_map_index = state_idx * feat_map_col + prob_idx;

    // build shared mems, dynamic
    extern __shared__ char shared_mems[];                // malloc share mem for each block
    float *d_feat_map_shared = (float*)shared_mems;      // cuda malloc for the 1st region
    int *d_ids_counter_shared = (int*)&d_feat_map_shared[feat_map_col];  // cuda malloc for the 2nd region, starts from the addr at the end of the 1st block


    //// reduce exp sum
    d_feat_map_shared[prob_idx] = exp(d_feat_map[feat_map_index]);
    __syncthreads();    //// sync among all threads within the current block, make sure d_feat_map_shared has valid exp vals within each block

    for (int step = 64; step > 0; step >>= 1) {
        if (prob_idx < step && (prob_idx + step) < feat_map_col) {
            d_feat_map_shared[prob_idx] += d_feat_map_shared[prob_idx + step];
        }
        __syncthreads();    //// wait all other block threads to reach here, make sure all threads have been updated
    }

    //// update exp sum for the 1st thread
    if (prob_idx == 0) {
        d_exp_val_sum[state_idx] = d_feat_map_shared[0];
    }


    //// reduce max element extraction, update shared mem again first
    d_feat_map_shared[prob_idx] = d_feat_map[feat_map_index];
    d_ids_counter_shared[prob_idx] = prob_idx;
    __syncthreads();

    float pivot_val, comp_val;
    int pivot_feature_idx, comp_feature_idx;
    int need_replace_flag;
    int d_state_id;
    float d_max_val_temp;

    for (int step = 64; step > 0; step >>= 1) {
        if (prob_idx < step && (prob_idx + step) < feat_map_col) {
            pivot_val = d_feat_map_shared[prob_idx];
            pivot_feature_idx = d_ids_counter_shared[prob_idx];

            comp_val = d_feat_map_shared[prob_idx + step];
            comp_feature_idx = d_ids_counter_shared[prob_idx + step];

            need_replace_flag = comp_val > pivot_val;

            // update feat map buffer
            d_feat_map_shared[prob_idx] = need_replace_flag * comp_val \
                + (1 - need_replace_flag) * pivot_val;

            // update loop ids counter
            d_ids_counter_shared[prob_idx] = need_replace_flag * comp_feature_idx \
                + (1 - need_replace_flag) * pivot_feature_idx;

            // get the max prob
            d_max_val_temp = d_feat_map_shared[prob_idx];

            // get the max index
            d_state_id = d_ids_counter_shared[prob_idx];
        }

        // make sure all threads within the same block have complete the current step work,
        // and then to the next step
        __syncthreads();
    }

    if (prob_idx == 0) {
        d_max_vals[state_idx] = d_max_val_temp;
        d_state_ids[state_idx] = d_state_id;
    }

}


void max_ele_extract_reduce_shared_mem_cuda(const int batch_size, 
    const int feat_map_row, const int feat_map_col, 
    float *d_feat_map, 
    int *d_state_ids, float *d_max_vals, float *d_exp_val_sum, 
    cudaStream_t &stream) {
    
    // define threads and block size, 2d map
    // here we know the featmap col is 95, so define size <feat_map_row, 64>
    max_ele_extract_reduce_shared_mem_kernel<<<feat_map_row * batch_size, feat_map_col, feat_map_col * sizeof(float) + feat_map_col * sizeof(int), stream>>>(
        batch_size, 
        feat_map_row, 
        feat_map_col, 
        static_cast<float*>(d_feat_map), 
        static_cast<int*>(d_state_ids), 
        static_cast<float*>(d_max_vals), 
        static_cast<float*>(d_exp_val_sum)
    );
}


/*
 * resize op using bilinear interpolation,
 *
 */
__global__ void resize_kernel(
                        const int batch_size,
                        const float *d_in, 
                        const int in_h, 
                        const int in_w,
                        const int channel_num, 
                        float rheight, 
                        float rwidth,
                        float *d_out, 
                        const int out_h, 
                        const int out_w, 
                        bool align_corners
                    ) {

    // 2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    const int yIndex = threadY % out_h;
    const int batch_id = threadY / out_h;

    // check border
    if(xIndex >= out_w || threadY >= out_h * batch_size) {
        return;
    }

    // input and output data byte size for each batch
    const size_t in_pannel = channel_num * in_h * in_w;
    const size_t out_pannel = out_h * out_w * channel_num;
    
    // same size, just assign values in target order
    if(in_h == out_h && in_w == out_w) 
    {
        for(int c = 0; c < channel_num; c ++) 
        {
            //// data_in order: NHWC
            //// data_out order: NCHW
            const int idx_in_batch = (yIndex * in_w + xIndex) * channel_num + c;
            const int idx_out_batch = c * out_h * out_w + yIndex * out_w + xIndex;
            d_out[batch_id * out_pannel + idx_out_batch] = d_in[batch_id * in_pannel + idx_in_batch];
        }
    }

    else 
    {
        // biliner interpolation, find position neighbors in d_in based on the thread idx
        //// compute source yIndex and its neighbors' yIndex, also including lambdas
        float y_src = area_pixel_compute_source_index<float>(rheight, yIndex, align_corners, /*cubic=*/false);
        const int y_low_src = y_src;
        const int y_high_src = y_low_src + ((y_low_src < in_h - 1) ? 1 : 0);
        const float lambda_y_low = static_cast<float>(y_src - y_low_src);
        const float lambda_y_high = static_cast<float>(1) - lambda_y_low;

        //// compute source xIndex and its neighbors' xIndex, also including lambdas
        float x_src = area_pixel_compute_source_index<float>(rwidth, xIndex, align_corners, /*cubic=*/false);
        const int x_low_src = x_src;
        const int x_high_src = x_low_src + ((x_low_src < in_w - 1) ? 1 : 0);
        const float lambda_x_low = static_cast<float>(x_src - x_low_src);
        const float lambda_x_high = static_cast<float>(1) - lambda_x_low;

        //// other useful vars
        for (int c = 0; c < channel_num; c ++) {
            ////// determine the out data index, NCHW order
            const int outIdx_batch = c * out_h * out_w + yIndex * out_w + xIndex;
            
            ////// interpolation, d_in is NHWC order
            float val = lambda_y_high * (
                    lambda_x_high * d_in[batch_id * in_pannel + y_low_src * in_w * channel_num + x_low_src * channel_num + c] + \
                    lambda_x_low * d_in[batch_id * in_pannel + y_low_src * in_w * channel_num + x_high_src * channel_num + c]
                ) + \
                lambda_y_low * (
                    lambda_x_high * d_in[batch_id * in_pannel + y_high_src * in_w * channel_num + x_low_src * channel_num + c] + \
                    lambda_x_low * d_in[batch_id * in_pannel + y_high_src * in_w * channel_num + x_high_src * channel_num + c]
                );

            ////// assign res
            d_out[batch_id * out_pannel + outIdx_batch] = static_cast<float>(val);
        }
    }
}
 

void resize_cuda(const int batch_size, 
    const int input_row, const int input_col, 
    const int output_row, const int output_col, 
    const int channel_num, const bool align_corners, 
    float *d_out, float *d_in,
    cudaStream_t &stream) {
    
    dim3 block(16, 16);
    const int grid_x = (output_col + block.x - 1) / block.x;
    const int grid_y = (output_row * batch_size + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    // compute height and width ratio, in_size / out_size
    float ratio_h = area_pixel_compute_scale<float>(input_row, output_row, align_corners);
    float ratio_w = area_pixel_compute_scale<float>(input_col, output_col, align_corners);
    
    resize_kernel<<<grid, block, 0, stream>>>(
        batch_size,
        static_cast<const float* const>(d_in), 
        input_row, input_col, 
        channel_num,
        ratio_h, ratio_w,
        static_cast<float*>(d_out), 
        output_row, output_col, align_corners
    );
}

 
/*
 * the center-aligned padding kernel
 * no normalization and channels flip, but will do NHWC -> NCHW
 */
 __global__ void preprocess_center_aligned_padding_kernel(
                                const int batch_size,
                                const int input_row, 
                                const int input_col, 
                                const int output_row, 
                                const int output_col, 
                                const int channel_num, 
                                const uint8_t padding_val, 
                                uint8_t *d_out, 
                                const uint8_t* const d_in
                            ) {
    // 2D Index of current thread
    const int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    if(threadX >= output_col || threadY >= output_row * batch_size) {
        return;
    }

    // calculate batch id and yIndex wrt each single batch
    const int batch_id = static_cast<int>(threadY / output_row);
    const int yIndex = threadY % output_row;
    const int batch_pixel_num_in = input_col * input_row * channel_num;
    const int batch_pixel_num_out = output_col * output_row * channel_num;

    // center-aligned padding
    const int offset_col = (output_col > input_col) ? (output_col - input_col) / 2 : 0;
    const int offset_row = (output_row > input_row) ? (output_row - input_row) / 2 : 0;
    
    for (int c = 0; c < channel_num; c ++) {
        //// NCHW order index
        const int index_out = batch_id * batch_pixel_num_out + threadX + yIndex * output_col + c * output_col * output_row;
        
        //// no need padding region, locate within the input data buffer
        if (threadX >= offset_col && threadX < offset_col + input_col &&
                yIndex >= offset_row && yIndex < offset_row + input_row) 
        {
            //// MHWC order index
            const int index_in = batch_id * batch_pixel_num_in + ( (threadX - offset_col) + (yIndex - offset_row) * input_col ) * channel_num;
            d_out[index_out] = d_in[index_in + c];
        } 

        //// padding region, outside the input buffer
        else 
        {   
            d_out[index_out] = padding_val;
        }
    }
}


void center_aligned_padding_cuda(const int batch_size, 
        const int input_row, const int input_col, 
        const int output_row, const int output_col, 
        const int channel_num, const uint8_t padding_val, 
        uint8_t *d_out, uint8_t *d_in, 
        cudaStream_t &stream) {
    
    // define block size based on the output size
    dim3 block(16, 16);
    const int grid_x = (output_col - 1) / block.x + 1;
    const int grid_y = (output_row * batch_size - 1) / block.y + 1;
    dim3 grid(grid_x, grid_y);

    preprocess_center_aligned_padding_kernel<<<grid, block, 0, stream>>>(
        batch_size,
        input_row, input_col, output_row, output_col, 
        channel_num, padding_val, 
        static_cast<uint8_t*>(d_out),
        static_cast<const uint8_t* const>(d_in)
    );
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
