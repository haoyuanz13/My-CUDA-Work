#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <assert.h>
#include "kernel.h"

#define MEAN_VAL_R 0.5
#define MEAN_VAL_G 0.3
#define MEAN_VAL_B 0.1
#define STD_VAL_R 0.5
#define STD_VAL_G 0.4
#define STD_VAL_B 0.8


/*
  check the Mat type
*/
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans + '0');
  return r;
}


/*
    main section
*/
int main(int argc, char* argv[]) {
    // init params, including demo data size, mean and std vals
    int data_width = 1024;
    int data_height = 720;
    int batch_size = 1;
    std::vector<float> mean_vals_rgb;
    std::vector<float> std_vals_frac_rgb;
    mean_vals_rgb.push_back(MEAN_VAL_R);
    mean_vals_rgb.push_back(MEAN_VAL_G);
    mean_vals_rgb.push_back(MEAN_VAL_B);
    std_vals_frac_rgb.push_back(1 / float(STD_VAL_R));
    std_vals_frac_rgb.push_back(1 / float(STD_VAL_G));
    std_vals_frac_rgb.push_back(1 / float(STD_VAL_B));

    // build randm mat with fixed size
    cv::Mat demo_data(data_height, data_width, CV_8UC3);
    cv::randu(demo_data, 0, 255);
    std::cout << "== demo data size (W x H): " << demo_data.size() << std::endl;
    std::cout << "== demo data channel num: " << demo_data.channels() << std::endl;
    std::cout << "== demo data type: " << type2str(demo_data.type()).c_str() << std::endl;

    // test loop number
    int loop_times = 21;

    // init time counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();


    /*
        cpu version, for loop, O(MN*3)
    */
    std::cout << "\n- CPU rgb image normalization and channel transfer ... " << std::endl;
    cv::Mat demo_data_processed_cpu(data_height, data_width, CV_32FC3);
    
    float time_process_sum_cpu = 0.0;
    for (int i = 0; i < loop_times; i ++) {
        startTime = std::chrono::high_resolution_clock::now();
        for (int row = 0; row < data_height; row ++) {
            for (int col = 0; col < data_width; col ++) {
                uint8_t pre_val_b = demo_data.at<cv::Vec3b>(row, col)[0];
                uint8_t pre_val_g = demo_data.at<cv::Vec3b>(row, col)[1];
                uint8_t pre_val_r = demo_data.at<cv::Vec3b>(row, col)[2];

                demo_data_processed_cpu.at<cv::Vec3f>(row, col)[0] = float(((pre_val_r / 255.0) - mean_vals_rgb[0]) * std_vals_frac_rgb[0]);
                demo_data_processed_cpu.at<cv::Vec3f>(row, col)[1] = float(((pre_val_g / 255.0) - mean_vals_rgb[1]) * std_vals_frac_rgb[1]);
                demo_data_processed_cpu.at<cv::Vec3f>(row, col)[2] = float(((pre_val_b / 255.0) - mean_vals_rgb[2]) * std_vals_frac_rgb[2]);
            }
        }
        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cpu = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_cpu += (i > 0)? time_process_iter_cpu : 0;    // remove the 1st time mem issue
    
    }
    printf("---> cpu process avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_cpu / float(loop_times - 1));
    std::cout << "== processed output size (W x H): " << demo_data_processed_cpu.size() << std::endl;
    std::cout << "== processed output channel num: " << demo_data_processed_cpu.channels() << std::endl;
    std::cout << "== processed output type: " << type2str(demo_data_processed_cpu.type()).c_str() << std::endl;


    /*
        GPU version
    */
    std::cout << "\n- CUDA rgb image normalization and channel transfer ... " << std::endl;
    cv::Mat demo_data_r_processed_cuda(data_height, data_width, CV_32FC1);
    cv::Mat demo_data_g_processed_cuda(data_height, data_width, CV_32FC1);
    cv::Mat demo_data_b_processed_cuda(data_height, data_width, CV_32FC1);

    // set gpu id and check cuda env
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));

    // create cudastream, here we use at most 2 streams
    cudaStream_t streams[2];
    for(auto&e : streams) cudaCheckError(cudaStreamCreate(&e));  // cuda stream
    cudaCheckError(cudaProfilerStart());

    // create pointers
    uint8_t *d_in;              // input demo data, d means device(gpu)
    float *d_out;               // output demo data which has been normalized
    void *d_mean;               // pointer to the mean value
    void *d_std_frac;           // use the std fraction to speed up kernel

    // malloc gpu mems
    cudaCheckError(cudaMalloc((void**)&d_in, sizeof(uint8_t) * data_height * data_width * 3 * batch_size));
    cudaCheckError(cudaMalloc((void**)&d_out, sizeof(float) * data_height * data_width * 3 * batch_size));
    cudaCheckError(cudaMalloc(&d_mean, sizeof(float) * 3));
    cudaCheckError(cudaMalloc(&d_std_frac, sizeof(float) * 3));

    cudaCheckError(cudaMemcpy(d_mean, mean_vals_rgb.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_std_frac, std_vals_frac_rgb.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));

    // exe kernel, for loop to make the result more reasonable
    float time_memcpy_h2d_sum = 0.0;
    float time_process_sum_cuda = 0.0;
    float time_memcpy_d2h_sum = 0.0;
    for (int i = 0; i < loop_times; i ++) {
        startTime = std::chrono::high_resolution_clock::now();
        assert(demo_data.isContinuous());

        cudaCheckError(cudaMemcpyAsync(
            d_in, demo_data.data, 
            sizeof(uint8_t) * data_height * data_width * 3, 
            cudaMemcpyHostToDevice, 
            streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        // printf("---> memcpy [h2d] cost: %.6f ms\n", time_memcpy_h2d_iter);
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda kernel
        startTime = std::chrono::high_resolution_clock::now();

        normalize_bgr2rgb_cuda(batch_size, 
            data_height, data_width, 
            d_out, d_in, 
            d_mean, d_std_frac,
            streams[0]);
        _cudaDeviceSynchronize();
        
        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cuda = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        // printf("---> cuda process cost: %.6f ms\n", time_process_iter_cuda);
        time_process_sum_cuda += (i > 0)? time_process_iter_cuda : 0; 

        //// memcpy from device to the host to the verification
        startTime = std::chrono::high_resolution_clock::now();
        cudaCheckError(cudaMemcpyAsync(demo_data_r_processed_cuda.data, 
                                    d_out, 
                                    sizeof(float) * data_height * data_width,
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));

        cudaCheckError(cudaMemcpyAsync(demo_data_g_processed_cuda.data, 
                                    d_out + data_height * data_width, 
                                    sizeof(float) * data_height * data_width,
                                    cudaMemcpyDeviceToHost, 
                                    streams[1]));

        cudaCheckError(cudaMemcpyAsync(demo_data_b_processed_cuda.data, 
                                    d_out + data_height * data_width * 2, 
                                    sizeof(float) * data_height * data_width,
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_d2h_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        // printf("---> memcpy [d2h] cost: %.6f ms\n", time_memcpy_d2h_iter);
        // std::cout << "== each processed channel output size (W x H): " << demo_data_r_processed_cuda.size() << std::endl;
        // std::cout << "== each processed channel output type: " << type2str(demo_data_r_processed_cuda.type()).c_str() << std::endl;
        time_memcpy_d2h_sum += (i > 0)? time_memcpy_d2h_iter : 0;
    }

    printf("---> memcpy [h2d] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_h2d_sum / float(loop_times - 1));
    printf("---> cuda process avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_cuda / float(loop_times - 1));
    printf("---> memcpy [d2h] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_d2h_sum / float(loop_times - 1));


    // compare cuda and our common cpu implementation
    std::cout << "\n- Compare CPU and CUDA Process Results ..." << std::endl;
    float diff_sum = 0.0;
    for (int row = 0; row < data_height; row ++) {
        for (int col = 0; col < data_width; col ++) {
            diff_sum += abs(demo_data_processed_cpu.at<cv::Vec3f>(row, col)[0] - demo_data_r_processed_cuda.at<float>(row, col));
            diff_sum += abs(demo_data_processed_cpu.at<cv::Vec3f>(row, col)[1] - demo_data_g_processed_cuda.at<float>(row, col));
            diff_sum += abs(demo_data_processed_cpu.at<cv::Vec3f>(row, col)[2] - demo_data_b_processed_cuda.at<float>(row, col));
        }
    }
    printf("---> the cpu-cuda process diff sum is %.6f, avg diff is %.6f \n", diff_sum, diff_sum / float(data_height * data_width * 3));


    // free up mems, be better in order
    std::cout << "\n- free cuda malloced mems and streams ..." << std::endl;
    cudaCheckError(cudaFree(d_std_frac));
    cudaCheckError(cudaFree(d_mean));
    cudaCheckError(cudaFree(d_out));
    cudaCheckError(cudaFree(d_in));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "- the rgb image normalization and channel transfer demo is completed!" << std::endl;

    return 0;
}
