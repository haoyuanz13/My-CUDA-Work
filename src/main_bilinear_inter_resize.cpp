#include <iostream>
#include <chrono>
#include <vector>
#include <assert.h>

#include "utils.h"
#include "kernel.h"




/*
    main section
*/
int main(int argc, char* argv[]) {
    // init params, including demo data size, mean and std vals
    int data_width_in = 1024;
    int data_height_in = 720;
    int data_width_out = 2048;
    int data_height_out = 1440;
    int batch_size = 1;
    const int data_channel = 3;
    const bool align_corners = false;  // consist with opencv resize liner inter

    // build randm mat with fixed size
    cv::Mat demo_data_in(data_height_in, data_width_in, CV_32FC3/*CV_8UC3*/);  // make it float to improve the precision
    cv::randu(demo_data_in, 0, 255);
    std::cout << "== demo input data size (W x H): " << demo_data_in.size() << std::endl;
    std::cout << "== demo input data channel num: " << demo_data_in.channels() << std::endl;
    std::cout << "== demo input data type: " << type2str(demo_data_in.type()).c_str() << std::endl;

    // test loop number
    int loop_times = 21;

    // init time counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();


    /*
        opencv version, cpu
    */
    std::cout << "\n- Opencv bilinear-interpolation resize ... " << std::endl;
    cv::Mat demo_data_processed_opencv(data_height_out, data_width_out, CV_32FC3/*CV_8UC3*/);

    float time_process_sum_opencv = 0.0;
    for (int i = 0; i < loop_times; i ++) {
        startTime = std::chrono::high_resolution_clock::now();
        
        cv::resize(demo_data_in, 
            demo_data_processed_opencv, 
            cv::Size(data_width_out, data_height_out), 
            cv::INTER_LINEAR);

        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_opencv = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_opencv += (i > 0)? time_process_iter_opencv : 0;    // remove the 1st time mem issue
    }
    printf("---> opencv biliner interpolation resize avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_opencv / float(loop_times - 1));
    std::cout << "== resized output size (W x H): " << demo_data_processed_opencv.size() << std::endl;
    std::cout << "== resized output channel num: " << demo_data_processed_opencv.channels() << std::endl;
    std::cout << "== resized output type: " << type2str(demo_data_processed_opencv.type()).c_str() << std::endl;


    /*
        GPU version
    */
    std::cout << "\n- CUDA bilinear-interpolation resize ... " << std::endl;
    cv::Mat demo_data_0_processed_cuda(data_height_out, data_width_out, CV_32FC1);
    cv::Mat demo_data_1_processed_cuda(data_height_out, data_width_out, CV_32FC1);
    cv::Mat demo_data_2_processed_cuda(data_height_out, data_width_out, CV_32FC1);

    // set gpu id and check cuda env
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));

    // create cudastream, here we use at most 2 streams
    cudaStream_t streams[2];
    for(auto&e : streams) cudaCheckError(cudaStreamCreate(&e));  // cuda stream
    cudaCheckError(cudaProfilerStart());

    // create pointers
    float *d_in;              // input demo data, d means device(gpu)
    float *d_out;             // output demo data which has been resized

    // malloc gpu mems
    cudaCheckError(cudaMalloc((void**)&d_in, sizeof(float) * data_height_in * data_width_in * data_channel * batch_size));
    cudaCheckError(cudaMalloc((void**)&d_out, sizeof(float) * data_height_out * data_width_out * data_channel * batch_size));

    // exe kernel, for loop to make the result more reasonable
    float time_memcpy_h2d_sum = 0.0;
    float time_process_sum_cuda = 0.0;
    float time_memcpy_d2h_sum = 0.0;

    assert(demo_data_in.isContinuous());
    for (int i = 0; i < loop_times; i ++) {
        startTime = std::chrono::high_resolution_clock::now();
        
        cudaCheckError(cudaMemcpyAsync(
            d_in, demo_data_in.data, 
            sizeof(float) * data_height_in * data_width_in * data_channel, 
            cudaMemcpyHostToDevice, 
            streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        // printf("---> memcpy [h2d] cost: %.6f ms\n", time_memcpy_h2d_iter);
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda kernel
        startTime = std::chrono::high_resolution_clock::now();

        resize_cuda(batch_size, 
            data_height_in, data_width_in,
            data_height_out, data_width_out, 
            data_channel, align_corners, 
            d_out, d_in, 
            streams[0]);

        _cudaDeviceSynchronize();
        
        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cuda = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        // printf("---> cuda process cost: %.6f ms\n", time_process_iter_cuda);
        time_process_sum_cuda += (i > 0)? time_process_iter_cuda : 0; 

        //// memcpy from device to the host to the verification
        startTime = std::chrono::high_resolution_clock::now();
        cudaCheckError(cudaMemcpyAsync(demo_data_0_processed_cuda.data, 
                                    d_out, 
                                    sizeof(float) * data_height_out * data_width_out,
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));

        cudaCheckError(cudaMemcpyAsync(demo_data_1_processed_cuda.data, 
                                    d_out + data_height_out * data_width_out, 
                                    sizeof(float) * data_height_out * data_width_out,
                                    cudaMemcpyDeviceToHost, 
                                    streams[1]));

        cudaCheckError(cudaMemcpyAsync(demo_data_2_processed_cuda.data, 
                                    d_out + data_height_out * data_width_out * 2, 
                                    sizeof(float) * data_height_out * data_width_out,
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


    /*
        compare results betwwen opencv and speed-up cuda implementation
    */
    std::cout << "\n- Compare Opencv and CUDA Resize Results ..." << std::endl;
    float diff_sum = 0.0;
    for (int row = 0; row < data_height_out; row ++) {
        for (int col = 0; col < data_width_out; col ++) {
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3f>(row, col)[0] - demo_data_0_processed_cuda.at<float>(row, col));
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3f>(row, col)[1] - demo_data_1_processed_cuda.at<float>(row, col));
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3f>(row, col)[2] - demo_data_2_processed_cuda.at<float>(row, col));
        }
    }
    printf("---> the [cuda] vs [opencv] res diff sum is %.6f, avg diff is %.6f \n", diff_sum, diff_sum / float(data_height_out * data_width_out * 3));


    /*
        free up mems, be better in order
    */
    std::cout << "\n- free cuda malloced mems and streams ..." << std::endl;
    cudaCheckError(cudaFree(d_out));
    cudaCheckError(cudaFree(d_in));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "- the bilinear-interpolation resize demo is completed!" << std::endl;

    return 0;
}
