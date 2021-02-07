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
    const int data_dst_width = 1024;
    const int data_dst_height = 720;
    const int data_roi_width = 1000;
    const int data_roi_height = 640;
    const int data_channel = 3;
    const int offset_width_indx = 10;
    const int offset_height_indx = 48;


    // build randm mat with fixed size
    cv::Mat demo_data_dst(data_dst_height, data_dst_width, CV_8UC3);
    cv::randu(demo_data_dst, 0, 255);
    std::cout << "== demo dst data size (W x H): " << demo_data_dst.size() << std::endl;
    std::cout << "== demo dst data channel num: " << demo_data_dst.channels() << std::endl;
    std::cout << "== demo dst data type: " << type2str(demo_data_dst.type()).c_str() << std::endl;

    cv::Mat demo_data_roi(data_roi_height, data_roi_width, CV_8UC3);
    cv::randu(demo_data_roi, 0, 255);
    std::cout << "== demo roi data size (W x H): " << demo_data_roi.size() << std::endl;
    std::cout << "== demo roi data channel num: " << demo_data_roi.channels() << std::endl;
    std::cout << "== demo roi data type: " << type2str(demo_data_roi.type()).c_str() << std::endl;

    // test loop number
    int loop_times = 21;

    // init time counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();


    /*
        cpu version, for loop, O(MN*3)
    */
    std::cout << "\n- CPU for-loop copyto ... " << std::endl;
    cv::Mat demo_data_processed_cpu(data_dst_height, data_dst_width, CV_8UC3);

    float time_process_sum_cpu = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();

        for (int row = 0; row < data_dst_height; row ++) 
        {
            for (int col = 0; col < data_dst_width; col ++) 
            {
                //// roi region
                if (row >= offset_height_indx && row < offset_height_indx + data_roi_height &&
                    col >= offset_width_indx && col < offset_width_indx + data_roi_width) 
                {
                    demo_data_processed_cpu.at<cv::Vec3b>(row, col)[0] = demo_data_roi.at<cv::Vec3b>(row - offset_height_indx, col - offset_width_indx)[0];
                    demo_data_processed_cpu.at<cv::Vec3b>(row, col)[1] = demo_data_roi.at<cv::Vec3b>(row - offset_height_indx, col - offset_width_indx)[1];
                    demo_data_processed_cpu.at<cv::Vec3b>(row, col)[2] = demo_data_roi.at<cv::Vec3b>(row - offset_height_indx, col - offset_width_indx)[2];
                }
                
                //// no roi region
                else 
                {

                    demo_data_processed_cpu.at<cv::Vec3b>(row, col)[0] = demo_data_dst.at<cv::Vec3b>(row, col)[0];
                    demo_data_processed_cpu.at<cv::Vec3b>(row, col)[1] = demo_data_dst.at<cv::Vec3b>(row, col)[1];
                    demo_data_processed_cpu.at<cv::Vec3b>(row, col)[2] = demo_data_dst.at<cv::Vec3b>(row, col)[2];
                }
            }
        }

        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cpu = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_cpu += (i > 0)? time_process_iter_cpu : 0;    // remove the 1st time mem issue
    
    }
    printf("---> cpu copyto avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_cpu / float(loop_times - 1));
    std::cout << "== copyto output size (W x H): " << demo_data_processed_cpu.size() << std::endl;
    std::cout << "== copyto output channel num: " << demo_data_processed_cpu.channels() << std::endl;
    std::cout << "== copyto output type: " << type2str(demo_data_processed_cpu.type()).c_str() << std::endl;


    /*
        opencv version, cpu
    */
    std::cout << "\n- Opencv copyto ... " << std::endl;
    cv::Mat demo_data_processed_opencv = demo_data_dst.clone();

    float time_process_sum_opencv = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        
        demo_data_roi.copyTo(demo_data_processed_opencv(
                cv::Rect(offset_width_indx, offset_height_indx, data_roi_width, data_roi_height)));

        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_opencv = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_opencv += (i > 0)? time_process_iter_opencv : 0;    // remove the 1st time mem issue
    }
    printf("---> opencv copyto avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_opencv / float(loop_times - 1));
    std::cout << "== padded output size (W x H): " << demo_data_processed_opencv.size() << std::endl;
    std::cout << "== padded output channel num: " << demo_data_processed_opencv.channels() << std::endl;
    std::cout << "== padded output type: " << type2str(demo_data_processed_opencv.type()).c_str() << std::endl;


    /*
        GPU version
    */
    std::cout << "\n- CUDA copyto ... " << std::endl;
    cv::Mat demo_data_0_processed_cuda(data_dst_height, data_dst_width, CV_8UC1);
    cv::Mat demo_data_1_processed_cuda(data_dst_height, data_dst_width, CV_8UC1);
    cv::Mat demo_data_2_processed_cuda(data_dst_height, data_dst_width, CV_8UC1);

    // set gpu id and check cuda env
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));

    // create cudastream, here we use at most 2 streams
    cudaStream_t streams[2];
    for(auto&e : streams) cudaCheckError(cudaStreamCreate(&e));  // cuda stream
    cudaCheckError(cudaProfilerStart());

    // create pointers
    uint8_t *d_src;             // input demo data, d means device(gpu)
    uint8_t *d_roi;             // input roi data
    uint8_t *d_out;             // output copied data

    // malloc gpu mems
    cudaCheckError(cudaMalloc((void**)&d_src, sizeof(uint8_t) * data_dst_height * data_dst_width * data_channel));
    cudaCheckError(cudaMalloc((void**)&d_roi, sizeof(uint8_t) * data_roi_height * data_roi_width * data_channel));
    cudaCheckError(cudaMalloc((void**)&d_out, sizeof(uint8_t) * data_dst_height * data_dst_width * data_channel));

    // exe kernel, for loop to make the result more reasonable
    float time_memcpy_h2d_sum = 0.0;
    float time_process_sum_cuda = 0.0;
    float time_memcpy_d2h_sum = 0.0;

    assert(demo_data_dst.isContinuous());
    assert(demo_data_roi.isContinuous());
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        
        cudaCheckError(cudaMemcpyAsync(
            d_src, demo_data_dst.data, 
            sizeof(uint8_t) * data_dst_height * data_dst_width * data_channel, 
            cudaMemcpyHostToDevice, 
            streams[0]));

        cudaCheckError(cudaMemcpyAsync(
            d_roi, demo_data_roi.data, 
            sizeof(uint8_t) * data_roi_height * data_roi_width * data_channel, 
            cudaMemcpyHostToDevice, 
            streams[1]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        // printf("---> memcpy [h2d] cost: %.6f ms\n", time_memcpy_h2d_iter);
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda kernel
        startTime = std::chrono::high_resolution_clock::now();

        copyto_hwc2chw_cuda(data_dst_height, data_dst_width,
            data_roi_height, data_roi_width, 
            data_channel, 
            offset_height_indx, offset_width_indx, 
            d_out, d_src, d_roi, 
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
                                    sizeof(uint8_t) * data_dst_height * data_dst_width,
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));

        cudaCheckError(cudaMemcpyAsync(demo_data_1_processed_cuda.data, 
                                    d_out + data_dst_height * data_dst_width, 
                                    sizeof(uint8_t) * data_dst_height * data_dst_width,
                                    cudaMemcpyDeviceToHost, 
                                    streams[1]));

        cudaCheckError(cudaMemcpyAsync(demo_data_2_processed_cuda.data, 
                                    d_out + data_dst_height * data_dst_width * 2, 
                                    sizeof(uint8_t) * data_dst_height * data_dst_width,
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
        compare results betwwen opencv, our common cpu and speed-up cuda implementation
    */
    std::cout << "\n---------------------------------------------------------------------------------" << std::endl;
    std::cout << "- Compare Opencv and CPU copyto Results ..." << std::endl;
    float diff_sum = 0.0;
    for (int row = 0; row < data_dst_height; row ++) 
    {
        for (int col = 0; col < data_dst_width; col ++) 
        {
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3b>(row, col)[0] - demo_data_processed_cpu.at<cv::Vec3b>(row, col)[0]);
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3b>(row, col)[1] - demo_data_processed_cpu.at<cv::Vec3b>(row, col)[1]);
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3b>(row, col)[2] - demo_data_processed_cpu.at<cv::Vec3b>(row, col)[2]);
        }
    }
    printf("---> the [opencv] vs [cpu] res diff sum is %.6f, avg diff is %.6f \n", diff_sum, diff_sum / float(data_dst_height * data_dst_width * 3));

    std::cout << "\n- Compare CPU and CUDA copyto Results ..." << std::endl;
    diff_sum = 0.0;
    for (int row = 0; row < data_dst_height; row ++) 
    {
        for (int col = 0; col < data_dst_width; col ++) 
        {
            diff_sum += abs(demo_data_processed_cpu.at<cv::Vec3b>(row, col)[0] - demo_data_0_processed_cuda.at<uint8_t>(row, col));
            diff_sum += abs(demo_data_processed_cpu.at<cv::Vec3b>(row, col)[1] - demo_data_1_processed_cuda.at<uint8_t>(row, col));
            diff_sum += abs(demo_data_processed_cpu.at<cv::Vec3b>(row, col)[2] - demo_data_2_processed_cuda.at<uint8_t>(row, col));
        }
    }
    printf("---> the [cpu] vs [cuda] res diff sum is %.6f, avg diff is %.6f \n", diff_sum, diff_sum / float(data_dst_height * data_dst_width * 3));

    std::cout << "\n- Compare Opencv and CUDA copyto Results ..." << std::endl;
    diff_sum = 0.0;
    for (int row = 0; row < data_dst_height; row ++) 
    {
        for (int col = 0; col < data_dst_width; col ++) 
        {
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3b>(row, col)[0] - demo_data_0_processed_cuda.at<uint8_t>(row, col));
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3b>(row, col)[1] - demo_data_1_processed_cuda.at<uint8_t>(row, col));
            diff_sum += abs(demo_data_processed_opencv.at<cv::Vec3b>(row, col)[2] - demo_data_2_processed_cuda.at<uint8_t>(row, col));
        }
    }
    printf("---> the [cuda] vs [opencv] res diff sum is %.6f, avg diff is %.6f \n", diff_sum, diff_sum / float(data_dst_height * data_dst_width * 3));


    /*
        free up mems, be better in order
    */
    std::cout << "\n- free cuda malloced mems and streams ..." << std::endl;
    cudaCheckError(cudaFree(d_out));
    cudaCheckError(cudaFree(d_roi));
    cudaCheckError(cudaFree(d_src));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "- the copyto demo is completed!" << std::endl;

    return 0;
}
