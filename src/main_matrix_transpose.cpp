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
    // init mat
    const int mat_row = 2048;
    const int mat_col = 4096;

    //// rand assign vals
    cv::Mat mat_demo(mat_row, mat_col, CV_32FC1);
    cv::randu(mat_demo, 0, 10);
    std::cout << "== demo mat size (W x H): " << mat_demo.size() << std::endl;
    std::cout << "== demo mat type: " << type2str(mat_demo.type()).c_str() << std::endl;

    // test loop number
    int loop_times = 21;

    // init time counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();


    /*
        cpu version, for loop, mat transpose
    */
    std::cout << "\n- CPU -- Mat Transpose ... " << std::endl;
    cv::Mat mat_T_cpu(mat_col, mat_row, CV_32FC1);
    float time_trans_sum_cpu = 0.0;
    for (int i = 0; i < loop_times; i ++) {
        startTime = std::chrono::high_resolution_clock::now();
        for (int row = 0; row < mat_row; row ++) 
        {
            for (int col = 0; col < mat_col; col ++) 
            {
                mat_T_cpu.at<float>(col, row) = mat_demo.at<float>(row, col);
            }
        }
        endTime = std::chrono::high_resolution_clock::now();
        float time_trans_iter_cpu = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_trans_sum_cpu += (i > 0)? time_trans_iter_cpu : 0;    // remove the 1st time mem issue
    }
    printf("---> cpu mat transpose avg cost(%d loops): %.6f ms\n", loop_times - 1, time_trans_sum_cpu / float(loop_times - 1));
    std::cout << "== mat transpose output size (W x H): " << mat_T_cpu.size() << std::endl;
    std::cout << "== mat transpose output type: " << type2str(mat_T_cpu.type()).c_str() << std::endl;


    /*
        use eigen to implement the mat transpose
    */
    std::cout << "\n- CPU -- Eigen Transpose ... " << std::endl;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_mat_demo;
    cv::cv2eigen(mat_demo, eigen_mat_demo);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_mat_T;
    
    float time_trans_sum_eigen = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        eigen_mat_T = eigen_mat_demo.transpose();
        endTime = std::chrono::high_resolution_clock::now();
        float time_trans_iter_eigen = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_trans_sum_eigen += (i > 0)? time_trans_iter_eigen : 0;    // remove the 1st time mem issue
    }
    printf("---> eigen mat transpose avg cost(%d loops): %.6f ms\n", loop_times - 1, time_trans_sum_eigen / float(loop_times - 1));
    std::cout << "== mat transpose output size (W x H): [" << eigen_mat_T.cols() << " x " << eigen_mat_T.rows() << "]" << std::endl;


    /*
        GPU version
    */
    std::cout << "\n- CUDA -- Mat Transpose ... " << std::endl;
    cv::Mat mat_T_cuda(mat_col, mat_row, CV_32FC1);

    // set gpu id and check cuda env
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));

    // create cudastream, here we use at most 2 streams
    cudaStream_t streams[2];
    for(auto&e : streams) cudaCheckError(cudaStreamCreate(&e));  // cuda stream
    cudaCheckError(cudaProfilerStart());

    // create pointers
    float *d_mat;     // float pointer to the input mat
    float *d_out;     // float pointer to the output mat C

    // malloc gpu mems
    cudaCheckError(cudaMalloc((void**)&d_mat, sizeof(float) * mat_row * mat_col));
    cudaCheckError(cudaMalloc((void**)&d_out, sizeof(float) * mat_col * mat_row));

    // check data store continuous
    assert(mat_demo.isContinuous());

    // exe kernel, for loop to make the result more reasonable
    float time_memcpy_h2d_sum = 0.0;
    float time_transpose_sum = 0.0;
    float time_memcpy_d2h_sum = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();

        cudaCheckError(cudaMemcpyAsync(
            d_mat, mat_demo.data, 
            mat_row * mat_col * sizeof(float), 
            cudaMemcpyHostToDevice, 
            streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda mat transpose
        startTime = std::chrono::high_resolution_clock::now();

        // mat_transpose_cuda(
        //     mat_row, mat_col, 
        //     d_mat, d_out,
        //     streams[0]);    // use global mem

        mat_transpose_shared_mem_naive_cuda(
            mat_row, mat_col, 
            d_mat, d_out,
            streams[0]);    // use shared mem to speed up, naive method

        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_transpose_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_transpose_sum += (i > 0)? time_transpose_iter : 0; 


        //// memcpy from device to the host to the verification
        startTime = std::chrono::high_resolution_clock::now();
        cudaCheckError(cudaMemcpyAsync(mat_T_cuda.data, d_out, 
                                    mat_col * mat_row * sizeof(float),
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_d2h_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_d2h_sum += (i > 0)? time_memcpy_d2h_iter : 0;
    }
    printf("---> memcpy [h2d] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_h2d_sum / float(loop_times - 1));
    printf("---> cuda matrix transpose avg cost(%d loops): %.6f ms\n", loop_times - 1, time_transpose_sum / float(loop_times - 1));
    printf("---> memcpy [d2h] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_d2h_sum / float(loop_times - 1));


    // compare matrix transpose res
    std::cout << "\n- Compare CPU - Eigen - CUDA Matrix transpose Results ..." << std::endl;
    float diff_sum_cpu_eigen = 0.0;
    float diff_sum_cpu_cuda = 0.0;
    for (int row = 0; row < mat_col; row ++) 
    {
        for (int col = 0; col < mat_row; col ++) 
        {
            diff_sum_cpu_eigen += abs(mat_T_cpu.at<float>(row, col) - eigen_mat_T(row, col));
            diff_sum_cpu_cuda += abs(mat_T_cpu.at<float>(row, col) - mat_T_cuda.at<float>(row, col));
        }
    }
    printf("---> the cpu-eigen transpose diff sum is %.6f, avg diff is %.6f \n", diff_sum_cpu_eigen, diff_sum_cpu_eigen / float(mat_col * mat_row));
    printf("---> the cpu-cuda transpose diff sum is %.6f, avg diff is %.6f \n", diff_sum_cpu_cuda, diff_sum_cpu_cuda / float(mat_col * mat_row));


    // free up mems, be better in order
    std::cout << "\n- free cuda malloced mems and streams ..." << std::endl;
    cudaCheckError(cudaFree(d_out));
    cudaCheckError(cudaFree(d_mat));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "- the matrix transpose demo is completed!" << std::endl;

    return 0;
}
