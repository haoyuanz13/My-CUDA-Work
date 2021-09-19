#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <assert.h>

#include "utils.h"
#include "kernel.h"


// char dict for this ocr reco scenario
std::vector<char> characters_dict;


/* decode postprocess into final ocr string */
void states_decoder_compare(const std::vector<int> &states_res_1, const std::vector<float> &probs_res_1, 
    const std::vector<int> &states_res_2, const std::vector<float> &probs_res_2, 
    const int feat_map_height, const int batch_size)
{
    // decode res holder
    std::string decoded_str_1 = "";
    std::string decoded_str_2 = "";
    float decoded_conf_sum_1 = 0.0;
    float decoded_conf_sum_2 = 0.0;

    // decode loop
    for (int ind = 0; ind < feat_map_height * batch_size; ind ++) 
    {
        //// new batch is here, clean up pre res
        if (ind % feat_map_height == 0)
        {
            decoded_str_1 = "";
            decoded_str_2 = "";
            decoded_conf_sum_1 = 0.0;
            decoded_conf_sum_2 = 0.0;
        }

        //// get conf and char, below conf 
        decoded_conf_sum_1 += probs_res_1[ind];
        decoded_conf_sum_2 += probs_res_2[ind];

        // blank char
        if (states_res_1[ind] != 0)
        {
            char char_ind = characters_dict[states_res_1[ind] - 1];
            decoded_str_1.push_back(char_ind);
        }

        if (states_res_2[ind] != 0)
        {
            char char_ind = characters_dict[states_res_2[ind] - 1];
            decoded_str_2.push_back(char_ind);
        }

        //// show compare res
        if ((ind + 1) % feat_map_height == 0)
        {
            int case_id = (ind + 1) / feat_map_height;
            int compare_state = decoded_str_1.compare(decoded_str_2);
            float conf_sum_diff = fabs(decoded_conf_sum_1 - decoded_conf_sum_2);

            printf("----> the %d case res -- str char diff number [%d] | conf sum diff [%.8f]\n", case_id, compare_state, conf_sum_diff);
        }
    }
}


/* main section */
int main(int argc, char* argv[]) 
{
    // init params, including demo data size, mean and std vals
    int feat_map_width = 95;     // ocr reco scenario, col number = char number
    int feat_map_height = 26;    // ocr reco scenario, row number = sequence number
    int batch_size = 32;

    // init chars dict
    characters_dict = 
    {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
        'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
        'W', 'X', 'Y', 'Z', '!', '\"', '#', '$', '%', '&', '\'', 
        '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', 
        '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'
    };

    // build randm mat with fixed size
    cv::Mat demo_feat_map(feat_map_height * batch_size, feat_map_width, CV_32FC1);  // make it float to improve the precision
    cv::randn(demo_feat_map, 1, 5);
    std::cout << "== demo feat map size (W x H): (" << feat_map_width << " x " << feat_map_height << ")" << std::endl;
    std::cout << "== demo feat map batch size: " << batch_size << std::endl;
    std::cout << "== demo feat map type: " << type2str(demo_feat_map.type()).c_str() << std::endl;

    // test loop number
    int loop_times = 21;
    
    // init time counter
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();


    /*
        cpu version, for loop to get each state id and metric info
    */
    std::cout << "\n- cpu brute-force postprocess ... " << std::endl;
    std::vector<int> states_res_cpu(feat_map_height * batch_size, -1);
    std::vector<float> probs_res_cpu(feat_map_height * batch_size, 0);

    float time_process_sum_cpu = 0.0;
    for (int i = 0; i < loop_times; i ++) 
    {
        startTime = std::chrono::high_resolution_clock::now();
        for (int row = 0; row < feat_map_height * batch_size; row ++)
        {
            // some helper vars
            float prob_exp_sum = 0.0;
            float max_prob_val = 0.0;
            int max_prob_col_index = -1;

            // traverse probs
            for (int col = 0; col < feat_map_width; col ++) 
            {
                float cur_prob = demo_feat_map.at<float>(row, col);
                prob_exp_sum += exp(cur_prob);

                // get flag
                int flag_need_update = cur_prob > max_prob_val;

                // compare and update
                max_prob_val = flag_need_update * cur_prob + (1 - flag_need_update) * max_prob_val;
                max_prob_col_index = flag_need_update * col + (1 - flag_need_update) * max_prob_col_index;
            }
            
            // update res for the current state
            states_res_cpu[row] = max_prob_col_index;
            probs_res_cpu[row] = exp(max_prob_val) / prob_exp_sum;
        }
        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cpu = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_cpu += (i > 0)? time_process_iter_cpu : 0;    // remove the 1st time mem issue
    }
    printf("---> cpu postprocess avg time cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_cpu / float(loop_times - 1));


    /*
        GPU version
    */
    std::cout << "\n- CUDA reduce postprocess ... " << std::endl;
    std::vector<int> states_res_cuda(feat_map_height * batch_size, -1);
    std::vector<float> probs_res_cuda(feat_map_height * batch_size, 0);

    // set gpu id and check cuda env
    const int gpu_id = 0;
    cudaCheckError(cudaSetDevice(gpu_id));

    // create cudastream, here we use at most 2 streams
    cudaStream_t streams[2];
    for(auto&e : streams) cudaCheckError(cudaStreamCreate(&e));  // cuda stream
    cudaCheckError(cudaProfilerStart());

    // create pointers
    float *d_in;              // input demo data, d means device(gpu)
    int *d_state_ids;         // output mem to store state ids
    float *d_max_vals;        // output mem to store max val
    float *d_exp_val_sum;     // output mem to store exp val sum for each state

    // malloc gpu mems
    cudaCheckError(cudaMalloc((void**)&d_in, sizeof(float) * feat_map_height * feat_map_width * 1 * batch_size));
    cudaCheckError(cudaMalloc((void**)&d_state_ids, sizeof(int) * 1 * feat_map_height * batch_size));
    cudaCheckError(cudaMalloc((void**)&d_max_vals, sizeof(float) * 1 * feat_map_height * batch_size));
    cudaCheckError(cudaMalloc((void**)&d_exp_val_sum, sizeof(float) * 1 * feat_map_height * batch_size));

    // exe kernel, for loop to make the result more reasonable
    float time_memcpy_h2d_sum = 0.0;
    float time_process_sum_cuda = 0.0;
    float time_memcpy_d2h_sum = 0.0;

    assert(demo_feat_map.isContinuous());
    for (int i = 0; i < loop_times; i ++) {
        startTime = std::chrono::high_resolution_clock::now();

        cudaCheckError(cudaMemcpyAsync(
            d_in, demo_feat_map.data, 
            sizeof(float) * feat_map_height * feat_map_width * 1 * batch_size, 
            cudaMemcpyHostToDevice, 
            streams[0]));
        _cudaDeviceSynchronize();

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_h2d_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_h2d_sum += (i > 0)? time_memcpy_h2d_iter : 0; 

        ////  cuda kernel
        startTime = std::chrono::high_resolution_clock::now();

        max_ele_extract_reduce_shared_mem_cuda(batch_size, 
            feat_map_height, feat_map_width, 
            d_in, 
            d_state_ids, 
            d_max_vals, 
            d_exp_val_sum, 
            streams[0]);
        _cudaDeviceSynchronize();
        
        endTime = std::chrono::high_resolution_clock::now();
        float time_process_iter_cuda = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_process_sum_cuda += (i > 0)? time_process_iter_cuda : 0; 

        //// memcpy from device to the host to the verification
        startTime = std::chrono::high_resolution_clock::now();

        float *h_max_vals = (float*)malloc(feat_map_height * batch_size * sizeof(float));
        float *h_exp_val_sum = (float*)malloc(feat_map_height * batch_size * sizeof(float));

        cudaCheckError(cudaMemcpyAsync(states_res_cuda.data(), d_state_ids, 
                                sizeof(int) * feat_map_height * batch_size,
                                cudaMemcpyDeviceToHost, 
                                streams[0]));
        
        cudaCheckError(cudaMemcpyAsync(h_max_vals, d_max_vals, 
                                sizeof(float) * feat_map_height * batch_size,
                                cudaMemcpyDeviceToHost, 
                                streams[1]));

        cudaCheckError(cudaMemcpyAsync(h_exp_val_sum, d_exp_val_sum, 
                                sizeof(float) * feat_map_height * batch_size,
                                cudaMemcpyDeviceToHost, 
                                streams[0]));

        _cudaDeviceSynchronize();

        for (int j = 0; j < feat_map_height * batch_size; j ++) 
        {
            probs_res_cuda[j] = exp(h_max_vals[j]) / h_exp_val_sum[j];
        }

        //// free cpu mem
        free(h_max_vals);
        free(h_exp_val_sum);

        endTime = std::chrono::high_resolution_clock::now();
        float time_memcpy_d2h_iter = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        time_memcpy_d2h_sum += (i > 0)? time_memcpy_d2h_iter : 0;
    }
    printf("---> memcpy [h2d] avg time cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_h2d_sum / float(loop_times - 1));
    printf("---> cuda process avg time cost(%d loops): %.6f ms\n", loop_times - 1, time_process_sum_cuda / float(loop_times - 1));
    printf("---> memcpy [d2h] avg time cost(%d loops): %.6f ms\n", loop_times - 1, time_memcpy_d2h_sum / float(loop_times - 1));


    /*
        compare results betwwen opencv and speed-up cuda implementation
    */
    std::cout << "\n- Compare CPU and CUDA PostProcess Results ..." << std::endl;
    states_decoder_compare(states_res_cpu, probs_res_cpu, states_res_cuda, probs_res_cuda, feat_map_height, batch_size);


    /*
        free up mems, be better in order
    */
    std::cout << "\n- free cuda malloced mems and streams ..." << std::endl;
    cudaCheckError(cudaFree(d_exp_val_sum));
    cudaCheckError(cudaFree(d_max_vals));
    cudaCheckError(cudaFree(d_state_ids));
    cudaCheckError(cudaFree(d_in));
    for(auto& e: streams) cudaStreamDestroy(e);
    cudaCheckError(cudaProfilerStop());

    std::cout << "\n- the reduce algorithm based cls postprocess demo is completed!" << std::endl;

    return 0;
}
