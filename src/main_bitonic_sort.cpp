#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <assert.h>

#include "utils.h"
#include "kernel.h"




/* show arr */
void printArr(int *arr, int len) 
{
    for (int i = 0; i < len; ++i) 
    {
        std::cout << std::setw(3) << arr[i];
    }

    std::cout << std::endl;
}

/* copy array */
void copyArr(int *arr_src, int *arr_dst, int len) 
{
    for (int i = 0; i < len; i ++) arr_dst[i] = arr_src[i];
}


/* main section */
int main(int argc, char* argv[]) 
{
    // build random int arr
    int len = 1024;  // limit by the block max thread number
    auto *arr = new int[len];

    std::srand((unsigned int) time(nullptr));
    for (int i = 0; i < len; ++i) 
    {
        arr[i] = (int) random() % 100;
    }
    // printArr(arr, len);


    // test loop number
    int loop_times = 21;

    // init time counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();


    /*
        cpu version, for loop
    */
    std::cout << "\n- CPU for loop bitonic sort ... " << std::endl;
    int *arr_cpu = new int[len];

    float time_process_sum_cpu = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        // reset arr cpy
        copyArr(arr, arr_cpu, len);

        // bitonic sort
        startTime = std::chrono::high_resolution_clock::now();

        for (int k = 2; k <= len; k *= 2) 
        {
            for (int j = k >> 1; j > 0; j = j >> 1) 
            {
                for (int i = 0; i < len; i ++) 
                {
                    int ixj = i ^ j;
                    if (ixj <= i) continue;
                    
                    if ( ((i & k) == 0) && (arr_cpu[i] > arr_cpu[ixj]) ) 
                    {
                        int tmp = arr_cpu[i];
                        arr_cpu[i] = arr_cpu[ixj];
                        arr_cpu[ixj] = tmp;
                    }

                    if ( ((i & k) != 0) && (arr_cpu[i] < arr_cpu[ixj]) ) 
                    {
                        int tmp = arr_cpu[i];
                        arr_cpu[i] = arr_cpu[ixj];
                        arr_cpu[ixj] = tmp;
                    }
                }
            }
        }

        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cpu = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_cpu += (i > 0)? time_process_iter_cpu : 0;    // remove the 1st time mem issue

    }
    printf("---> cpu bitonic avg cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_cpu / float(loop_times - 1));
    // printArr(arr_cpu, len);


    /*
        GPU version
    */
    std::cout << "\n- CUDA bitonic sort ... " << std::endl;
    int *arr_cuda = new int[len];

    // set gpu id and check cuda env
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));

    // create cudastream, here we use at most 2 streams
    cudaStream_t streams[2];
    for(auto&e : streams) cudaCheckError(cudaStreamCreate(&e));  // cuda stream
    cudaCheckError(cudaProfilerStart());

    // create pointers
    int *d_arr;             // input demo data, d means device(gpu)

    // malloc gpu mems
    cudaCheckError(cudaMalloc((void**)&d_arr, sizeof(int) * len));

    // exe kernel, for loop to make the result more reasonable
    float time_memcpy_h2d_sum = 0.0;
    float time_process_sum_cuda = 0.0;
    float time_memcpy_d2h_sum = 0.0;

    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        
        cudaCheckError(cudaMemcpyAsync(
            d_arr, arr, 
            sizeof(int) * len, 
            cudaMemcpyHostToDevice, 
            streams[0]));
            
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        // printf("---> memcpy [h2d] cost: %.6f ms\n", time_memcpy_h2d_iter);
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda kernel
        startTime = std::chrono::high_resolution_clock::now();

        bitonic_sort_cuda(len, d_arr, streams[0]);

        _cudaDeviceSynchronize();
        
        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cuda = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        // printf("---> cuda process cost: %.6f ms\n", time_process_iter_cuda);
        time_process_sum_cuda += (i > 0)? time_process_iter_cuda : 0; 

        //// memcpy from device to the host to the verification
        startTime = std::chrono::high_resolution_clock::now();
        cudaCheckError(cudaMemcpyAsync(arr_cuda, 
                                    d_arr, 
                                    sizeof(int) * len,
                                    cudaMemcpyDeviceToHost, 
                                    streams[0]));
        _cudaDeviceSynchronize();

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
    // printArr(arr_cuda, len);


    /*
        compare results betwwen common cpu and speed-up cuda implementation
    */
    std::cout << "\n---------------------------------------------------------------------------------" << std::endl;
    std::cout << "- Compare CUDA and CPU bitonic sort Results ..." << std::endl;
    float diff_sum = 0.0;
    for (int i = 0; i < len; i ++) 
    {
        diff_sum += abs(arr_cpu[i] - arr_cuda[i]);
    }
    printf("---> the [cpu] vs [cuda] res diff sum is %.6f, avg diff is %.6f \n", diff_sum, diff_sum / float(len));


    /*
        free up mems, be better in order
    */
    std::cout << "\n- free cpu and cuda malloced mems and streams ..." << std::endl;
    free(arr_cuda);
    free(arr_cpu);
    free(arr);

    cudaCheckError(cudaFree(d_arr));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "- the bitonic sort demo is completed!" << std::endl;

    return 0;
}
