#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cmath>
#include <assert.h>
#include "kernel.h"

#define MEAN_VAL 0.5
#define STD_VAL 0.5


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
    float mean_val = MEAN_VAL;
    float std_val_frac = 1 / float(STD_VAL);    // here we use fraction since the multiply is faster than division

    // build randm mat with fixed size
    cv::Mat demo_data(data_height, data_width, CV_8UC1);
    cv::randu(demo_data, 0, 255);
    std::cout << "== demo data size (W x H): " << demo_data.size() << std::endl;
    std::cout << "== demo data type: " << type2str(demo_data.type()).c_str() << std::endl;


    /*
        cpu version, for loop normalization, O(MN)
    */
    std::cout << "\n- CPU gray image normalization ... " << std::endl;
    cv::Mat demo_data_normed_cpu(data_height, data_width, CV_32FC1);
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < data_height; row ++) {
        for (int col = 0; col < data_width; col ++) {
            uint8_t pre_val = demo_data.at<uint8_t>(row, col);
            demo_data_normed_cpu.at<float>(row, col) = float(((pre_val / 255.0) - MEAN_VAL) / STD_VAL);
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    float time_normalization_iter_cpu = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    printf("---> cpu normalization cost: %.6f ms\n", time_normalization_iter_cpu);
    std::cout << "== normed output size (W x H): " << demo_data_normed_cpu.size() << std::endl;
    std::cout << "== normed output type: " << type2str(demo_data_normed_cpu.type()).c_str() << std::endl;

    /*
        GPU version
    */
    std::cout << "\n- CUDA gray image normalization ... " << std::endl;

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
    cudaCheckError(cudaMalloc((void**)&d_in, sizeof(uint8_t) * data_height * data_width * 1 * batch_size));
    cudaCheckError(cudaMalloc((void**)&d_out, sizeof(float) * data_height * data_width * 1 * batch_size));
    cudaCheckError(cudaMalloc(&d_mean, sizeof(float)));
    cudaCheckError(cudaMalloc(&d_std_frac, sizeof(float)));

    cudaCheckError(cudaMemcpy(d_mean, &mean_val, sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_std_frac, &std_val_frac, sizeof(float), cudaMemcpyHostToDevice));

    // exe kernel
    startTime = std::chrono::high_resolution_clock::now();
    assert(demo_data.isContinuous());

    uint8_t *h_demo_data = demo_data.ptr<uint8_t>(0);
    cudaCheckError(cudaMemcpyAsync(
        d_in, h_demo_data, 
        data_height * data_width * sizeof(uint8_t), 
        cudaMemcpyHostToDevice, 
        streams[0]));
    _cudaDeviceSynchronize();

    endTime = std::chrono::high_resolution_clock::now();
    float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    printf("---> memcoy [h2d] cost: %.6f ms\n", time_memcpy_h2d_iter);

    ////  cuda normalization
    startTime = std::chrono::high_resolution_clock::now();

    normalize_gray_cuda(batch_size, 
        data_height, data_width, 
        d_out, d_in, 
        d_mean, d_std_frac,
        streams[0]);
    _cudaDeviceSynchronize();
    
    endTime = std::chrono::high_resolution_clock::now();
    float time_normalization_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    printf("---> cuda normalization cost: %.6f ms\n", time_normalization_iter);

    //// memcpy from device to the host to the verification
    startTime = std::chrono::high_resolution_clock::now();
    cv::Mat demo_data_normed_cuda(data_height, data_width, CV_32FC1);
    cudaCheckError(cudaMemcpyAsync(demo_data_normed_cuda.data, 
                                d_out, 
                                sizeof(float) * data_height * data_width,
                                cudaMemcpyDeviceToHost, 
                                streams[0]
                            ));

    endTime = std::chrono::high_resolution_clock::now();
    float time_memcpy_d2h_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    printf("---> memcoy [d2h] cost: %.6f ms\n", time_memcpy_d2h_iter);
    std::cout << "== normed output size (W x H): " << demo_data_normed_cuda.size() << std::endl;
    std::cout << "== normed output type: " << type2str(demo_data_normed_cuda.type()).c_str() << std::endl;


    // compare cuda and our common cpu implementation
    std::cout << "\n- Compare CPU and CUDA Normalization Results ..." << std::endl;
    float diff_sum = 0.0;
    for (int row = 0; row < data_height; row ++) {
        for (int col = 0; col < data_width; col ++) {
            diff_sum += abs(demo_data_normed_cpu.at<float>(row, col) - demo_data_normed_cuda.at<float>(row, col));
        }
    }
    printf("---> the cpu-cuda normalization diff sum is %.6f, avg diff is %.6f \n", diff_sum, diff_sum / float(data_height * data_width));


    // free up mems, be better in order
    std::cout << "\n- free cuda malloced mems and streams ..." << std::endl;
    cudaCheckError(cudaFree(d_std_frac));
    cudaCheckError(cudaFree(d_mean));
    cudaCheckError(cudaFree(d_out));
    cudaCheckError(cudaFree(d_in));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "- the gray image cuda normalization demo is completed!" << std::endl;

    return 0;
}