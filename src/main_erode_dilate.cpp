#include <iostream>
#include <chrono>
#include <vector>
#include <assert.h>

#include "utils.h"
#include "kernel.h"
#include "morphology.h"




/*
    main section
*/
int main(int argc, char* argv[]) 
{
    // load demo bimap
    const std::string demo_bimap_dir = "./data/bimap_demo.jpg";
    cv::Mat demo_bimap_raw = cv::imread(demo_bimap_dir, cv::IMREAD_GRAYSCALE);
    std::cout << "== demo raw bimap size (W x H): " << demo_bimap_raw.size() << std::endl;
    std::cout << "== demo raw bimap channel num: " << demo_bimap_raw.channels() << std::endl;
    std::cout << "== demo raw bimap type: " << type2str(demo_bimap_raw.type()).c_str() << std::endl;

    // init params, including demo data size and kernel element size
    const int data_resized_w = 1024;
    const int data_resized_h = 720;
    const int elem_size_w = 17;
    const int elem_size_h = 17;
    const int radio_w = 8;  // (elem_size - 1) / 2
    const int radio_h = 8;

    // resize raw data
    cv::Mat demo_bimap_resized(data_resized_h, data_resized_w, CV_8UC1);
    cv::resize(demo_bimap_raw, 
        demo_bimap_resized, 
        cv::Size(data_resized_w, data_resized_h), 
        cv::INTER_LINEAR);

    std::cout << "== demo resized bimap size (W x H): " << demo_bimap_resized.size() << std::endl;
    std::cout << "== demo resized bimap channel num: " << demo_bimap_resized.channels() << std::endl;
    std::cout << "== demo resized bimap type: " << type2str(demo_bimap_resized.type()).c_str() << std::endl;

    // print element struct size
    printf("\n== demo struct element size H: %d | W: %d\n", elem_size_h, elem_size_w);


    // test loop number
    int loop_times = 21;

    // init time counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();

    /*
        opencv version, cpu
    */
    std::cout << "\n- Opencv erode and dilate ... " << std::endl;
    cv::Mat demo_bimap_erode_cpu = demo_bimap_resized.clone();
    cv::Mat elem_rect = cv::getStructuringElement(
        cv::MORPH_RECT, 
        cv::Size(elem_size_w, elem_size_h), 
        cv::Point(radio_w, radio_h));  // build kernel

    float time_process_sum_opencv = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        
        cv::erode(demo_bimap_resized, demo_bimap_erode_cpu, 
            elem_rect, cv::Point(radio_w, radio_h));

        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_opencv = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_opencv += (i > 0)? time_process_iter_opencv : 0;    // remove the 1st time mem issue
    }
    printf("---> opencv erode avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_opencv / float(loop_times - 1));

    cv::Mat demo_bimap_dilate_cpu = demo_bimap_resized.clone();
    time_process_sum_opencv = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        
        cv::dilate(demo_bimap_resized, demo_bimap_dilate_cpu, 
            elem_rect, cv::Point(radio_w, radio_h));

        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_opencv = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_opencv += (i > 0)? time_process_iter_opencv : 0;    // remove the 1st time mem issue
    }
    printf("---> opencv dilate avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_opencv / float(loop_times - 1));


    /*
        GPU version
    */
    std::cout << "\n- CUDA erode and dilate ... " << std::endl;
    cv::Mat demo_bimap_erode_cuda(data_resized_h, data_resized_w, CV_8UC1);
    cv::Mat demo_bimap_dilate_cuda(data_resized_h, data_resized_w, CV_8UC1);

    // set gpu id and check cuda env
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));

    // create cudastream, here we use at most 2 streams
    cudaStream_t streams[2];
    for(auto&e : streams) cudaCheckError(cudaStreamCreate(&e));  // cuda stream
    cudaCheckError(cudaProfilerStart());

    // create pointers
    uint8_t *d_resized;             // input demo data, d means device(gpu)
    uint8_t *d_temp;                // temp data for gpu usage
    uint8_t *d_out_erode;           // output for erode res
    uint8_t *d_out_dilate;          // output for dilate res

    // malloc gpu mems
    cudaCheckError(cudaMalloc((void**)&d_resized, sizeof(uint8_t) * data_resized_h * data_resized_w));
    cudaCheckError(cudaMalloc((void**)&d_temp, sizeof(uint8_t) * data_resized_h * data_resized_w));
    cudaCheckError(cudaMalloc((void**)&d_out_erode, sizeof(uint8_t) * data_resized_h * data_resized_w));
    cudaCheckError(cudaMalloc((void**)&d_out_dilate, sizeof(uint8_t) * data_resized_h * data_resized_w));

    // time counter
    float time_memcpy_h2d_sum = 0.0;
    float time_process_sum_cuda = 0.0;
    float time_memcpy_d2h_sum = 0.0;

    // exe kernel, for loop to make the result more reasonable
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        
        cudaCheckError(cudaMemcpyAsync(
            d_resized, demo_bimap_resized.data, 
            sizeof(uint8_t) * data_resized_h * data_resized_w, 
            cudaMemcpyHostToDevice, 
            streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda kernel
        startTime = std::chrono::high_resolution_clock::now();

        ErodeTwoStepShared(d_resized, d_temp, d_out_erode,
            radio_w, data_resized_w, data_resized_h, 
            streams[0]);

        _cudaDeviceSynchronize();
        
        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cuda = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_cuda += (i > 0)? time_process_iter_cuda : 0; 

        //// memcpy from device to the host to the verification
        startTime = std::chrono::high_resolution_clock::now();
        cudaCheckError(cudaMemcpyAsync(demo_bimap_erode_cuda.data, 
                                    d_out_erode, 
                                    sizeof(uint8_t) * data_resized_h * data_resized_w,
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_d2h_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_d2h_sum += (i > 0)? time_memcpy_d2h_iter : 0;
    }
    printf("---> erode memcpy [h2d] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_h2d_sum / float(loop_times - 1));
    printf("---> erode kernel process avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_cuda / float(loop_times - 1));
    printf("---> erode memcpy [d2h] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_d2h_sum / float(loop_times - 1));


    time_memcpy_h2d_sum = 0.0;
    time_process_sum_cuda = 0.0;
    time_memcpy_d2h_sum = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        
        cudaCheckError(cudaMemcpyAsync(
            d_resized, demo_bimap_resized.data, 
            sizeof(uint8_t) * data_resized_h * data_resized_w, 
            cudaMemcpyHostToDevice, 
            streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda kernel
        startTime = std::chrono::high_resolution_clock::now();

        DilateTwoStepShared(d_resized, d_temp, d_out_dilate,
            radio_w, data_resized_w, data_resized_h, 
            streams[0]);

        _cudaDeviceSynchronize();
        
        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cuda = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_cuda += (i > 0)? time_process_iter_cuda : 0; 

        //// memcpy from device to the host to the verification
        startTime = std::chrono::high_resolution_clock::now();
        cudaCheckError(cudaMemcpyAsync(demo_bimap_dilate_cuda.data, 
                                    d_out_dilate, 
                                    sizeof(uint8_t) * data_resized_h * data_resized_w,
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_d2h_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_d2h_sum += (i > 0)? time_memcpy_d2h_iter : 0;
    }
    printf("---> dilate memcpy [h2d] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_h2d_sum / float(loop_times - 1));
    printf("---> dilate kernel process avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_cuda / float(loop_times - 1));
    printf("---> dilate memcpy [d2h] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_d2h_sum / float(loop_times - 1));


    /*
        compare results betwwen opencv cpu and speed-up cuda implementation
    */
    std::cout << "\n- Compare Opencv and CUDA erode and dilate Results ..." << std::endl;
    float diff_sum_erode = 0.0;
    float diff_sum_dilate = 0.0;
    for (int row = 0; row < data_resized_h; row ++) 
    {
        for (int col = 0; col < data_resized_w; col ++) 
        {
            diff_sum_erode += abs(demo_bimap_erode_cpu.at<uint8_t>(row, col) - demo_bimap_erode_cuda.at<uint8_t>(row, col));
            diff_sum_dilate += abs(demo_bimap_dilate_cpu.at<uint8_t>(row, col) - demo_bimap_dilate_cuda.at<uint8_t>(row, col));
        }
    }
    printf("---> the [cuda] vs [opencv] erode res diff sum is %.6f, avg diff is %.6f \n", diff_sum_erode, diff_sum_erode / float(data_resized_h * data_resized_w));
    printf("---> the [cuda] vs [opencv] dilate res diff sum is %.6f, avg diff is %.6f \n", diff_sum_dilate, diff_sum_dilate / float(data_resized_h * data_resized_w));


    /*
        free up mems, be better in order
    */
    std::cout << "\n- free cuda malloced mems and streams ..." << std::endl;
    cudaCheckError(cudaFree(d_temp));
    cudaCheckError(cudaFree(d_out_dilate));
    cudaCheckError(cudaFree(d_out_erode));
    cudaCheckError(cudaFree(d_resized));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "- the erode-dilate demo is completed!" << std::endl;

    return 0;
}
