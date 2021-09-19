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
    const int mat_A_row = 512;
    const int mat_A_col = 512;
    const int mat_B_row = mat_A_col;
    const int mat_B_col = 512;

    //// rand assign vals
    cv::Mat mat_A(mat_A_row, mat_A_col, CV_32FC1);
    cv::randu(mat_A, 0, 10);
    std::cout << "== demo mat A size (W x H): " << mat_A.size() << std::endl;
    std::cout << "== demo mat A type: " << type2str(mat_A.type()).c_str() << std::endl;

    cv::Mat mat_B(mat_B_row, mat_B_col, CV_32FC1);
    cv::randu(mat_B, 0, 10);
    std::cout << "== demo mat B size (W x H): " << mat_B.size() << std::endl;
    std::cout << "== demo mat B type: " << type2str(mat_B.type()).c_str() << std::endl;

    // test loop number
    int loop_times = 21;

    // init time counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();


    /*
        cpu version, for loop, mat multiplication
    */
    std::cout << "\n- CPU -- Mat Multiplication ... " << std::endl;
    cv::Mat mat_AxB_cpu(mat_A_row, mat_B_col, CV_32FC1);
    float time_multi_sum_cpu = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        for (int row = 0; row < mat_A_row; row ++) 
        {
            for (int col = 0; col < mat_B_col; col ++) 
            {
                float sum = 0.0;
                for (int i = 0; i < mat_A_col; i ++)
                {
                    sum += mat_A.at<float>(row, i) * mat_B.at<float>(i, col);
                }
                mat_AxB_cpu.at<float>(row, col) = sum;
            }
        }
        endTime = std::chrono::high_resolution_clock::now();
        float time_multi_iter_cpu = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_multi_sum_cpu += (i > 0)? time_multi_iter_cpu : 0;    // remove the 1st time mem issue
    }
    printf("---> cpu mat multiplication avg cost(%d loops): %.6f ms\n", loop_times - 1, time_multi_sum_cpu / float(loop_times - 1));
    std::cout << "== mat multiplication output size (W x H): " << mat_AxB_cpu.size() << std::endl;


    /*
        use eigen to implement the mat multiplication
    */
    std::cout << "\n- CPU -- Eigen Multiplication ... " << std::endl;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_mat_A;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_mat_B;
    Eigen::MatrixXf eigen_multi_res;
    cv::cv2eigen(mat_A, eigen_mat_A);
    cv::cv2eigen(mat_B, eigen_mat_B);
    
    float time_multi_sum_eigen = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        eigen_multi_res = eigen_mat_A * eigen_mat_B;
        endTime = std::chrono::high_resolution_clock::now();
        float time_multi_iter_eigen = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_multi_sum_eigen += (i > 0)? time_multi_iter_eigen : 0;    // remove the 1st time mem issue
    }
    printf("---> eigen mat multiplication avg cost(%d loops): %.6f ms\n", loop_times - 1, time_multi_sum_eigen / float(loop_times - 1));
    std::cout << "== mat multiplication output size (W x H): [" << eigen_multi_res.cols() << " x " << eigen_multi_res.rows() << "]" << std::endl;


    /*
        GPU version
    */
    std::cout << "\n- CUDA -- Mat Multiplication ... " << std::endl;
    cv::Mat mat_AxB_cuda(mat_A_row, mat_B_col, CV_32FC1);

    // set gpu id and check cuda env
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));

    // create cudastream, here we use at most 2 streams
    cudaStream_t streams[2];
    for(auto&e : streams) cudaCheckError(cudaStreamCreate(&e));  // cuda stream
    cudaCheckError(cudaProfilerStart());

    // create pointers
    float *d_A;     // float pointer to the input mat A
    float *d_B;     // float pointer to the input mat B
    float *d_C;     // float pointer to the output mat C

    // malloc gpu mems
    cudaCheckError(cudaMalloc((void**)&d_A, sizeof(float) * mat_A_row * mat_A_col));
    cudaCheckError(cudaMalloc((void**)&d_B, sizeof(float) * mat_B_row * mat_B_col));
    cudaCheckError(cudaMalloc((void**)&d_C, sizeof(float) * mat_A_row * mat_B_col));

    // check data store continuous
    assert(mat_A.isContinuous());
    assert(mat_B.isContinuous());

    // exe kernel, for loop to make the result more reasonable
    float time_memcpy_h2d_sum = 0.0;
    float time_multiplication_sum = 0.0;
    float time_memcpy_d2h_sum = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();

        cudaCheckError(cudaMemcpyAsync(
            d_A, mat_A.data, 
            mat_A_row * mat_A_col * sizeof(float), 
            cudaMemcpyHostToDevice, 
            streams[0]));
        
        cudaCheckError(cudaMemcpyAsync(
            d_B, mat_B.data, 
            mat_B_row * mat_B_col * sizeof(float), 
            cudaMemcpyHostToDevice, 
            streams[1]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda mat multiplication
        startTime = std::chrono::high_resolution_clock::now();

        // mat_multiply_cuda(
        //     mat_A_row, mat_A_col, mat_B_col, 
        //     d_A, d_B, d_C, 
        //     streams[0]);    // use global mem

        mat_multiply_share_mem_cuda(
            mat_A_row, mat_A_col, mat_B_col, 
            d_A, d_B, d_C, 
            streams[0]);    // use shared mem to speed up

        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_multiplication_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_multiplication_sum += (i > 0)? time_multiplication_iter : 0; 


        //// memcpy from device to the host to the verification
        startTime = std::chrono::high_resolution_clock::now();
        cudaCheckError(cudaMemcpyAsync(mat_AxB_cuda.data, d_C, 
                                    mat_A_row * mat_B_col * sizeof(float),
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_d2h_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_d2h_sum += (i > 0)? time_memcpy_d2h_iter : 0;
    }
    printf("---> memcpy [h2d] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_h2d_sum / float(loop_times - 1));
    printf("---> cuda matrix multiplication avg cost(%d loops): %.6f ms\n", loop_times - 1, time_multiplication_sum / float(loop_times - 1));
    printf("---> memcpy [d2h] avg cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_d2h_sum / float(loop_times - 1));


    // compare matrix multiplication res
    std::cout << "\n- Compare CPU - Eigen - CUDA Matrix Multiplication Results ..." << std::endl;
    float diff_sum_cpu_eigen = 0.0;
    float diff_sum_cpu_cuda = 0.0;
    for (int row = 0; row < mat_A_row; row ++) 
    {
        for (int col = 0; col < mat_B_col; col ++) 
        {
            diff_sum_cpu_eigen += abs(mat_AxB_cpu.at<float>(row, col) - eigen_multi_res(row, col));
            diff_sum_cpu_cuda += abs(mat_AxB_cpu.at<float>(row, col) - mat_AxB_cuda.at<float>(row, col));
        }
    }
    printf("---> the cpu-eigen multi diff sum is %.6f, avg diff is %.6f \n", diff_sum_cpu_eigen, diff_sum_cpu_eigen / float(mat_A_row * mat_B_col));
    printf("---> the cpu-cuda multi diff sum is %.6f, avg diff is %.6f \n", diff_sum_cpu_cuda, diff_sum_cpu_cuda / float(mat_A_row * mat_B_col));


    // free up mems, be better in order
    std::cout << "\n- free cuda malloced mems and streams ..." << std::endl;
    cudaCheckError(cudaFree(d_C));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_A));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "- the matrix multiplication demo is completed!" << std::endl;

    return 0;
}
