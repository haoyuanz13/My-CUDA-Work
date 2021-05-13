#include "icp.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

// for matrix multi
#include "cublas_v2.h"

namespace cuda_icp 
{

// for debug info
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void transform_pcd_cuda(Vec3f* model_pcd_ptr, uint32_t model_pcd_size, Mat4x4f trans){
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= model_pcd_size) return;

    Vec3f& pcd = model_pcd_ptr[i];
    float new_x = trans[0][0]*pcd.x + trans[0][1]*pcd.y + trans[0][2]*pcd.z + trans[0][3];
    float new_y = trans[1][0]*pcd.x + trans[1][1]*pcd.y + trans[1][2]*pcd.z + trans[1][3];
    float new_z = trans[2][0]*pcd.x + trans[2][1]*pcd.y + trans[2][2]*pcd.z + trans[2][3];
    pcd.x = new_x;
    pcd.y = new_y;
    pcd.z = new_z;
}


template<class Scene>
RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f> &model_pcd, const Scene scene,
                                        const ICPConvergenceCriteria criteria) 
{
    RegistrationResult result;
    RegistrationResult backup;

    thrust::host_vector<float> A_host(36, 0);
    thrust::host_vector<float> b_host(6, 0);

    const uint32_t threadsPerBlock = 256;
    const uint32_t numBlocks = (model_pcd.size() + threadsPerBlock - 1) / threadsPerBlock;

    for(uint32_t iter = 0; iter <= criteria.max_iteration_; iter ++) 
    {
        Vec29f Ab_tight = thrust::transform_reduce(thrust::cuda::par.on(cudaStreamPerThread),
                                        model_pcd.begin_thr(), model_pcd.end_thr(), thrust__pcd2Ab<Scene>(scene),
                                        Vec29f::Zero(), thrust__plus());

        // method from icpcuda
//        Vec29f Ab_tight = custom_trans_reduce::transform_reduce(model_pcd.data(), model_pcd.size(),
//                                                                thrust__pcd2Ab<Scene>(scene), Vec29f::Zero());

        cudaStreamSynchronize(cudaStreamPerThread);
        backup = result;  //同步

        float& count = Ab_tight[28];  // 1有结果
        float& total_error = Ab_tight[27];
        if(count == 0) return result;  // avoid divid 0

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = std::sqrt(total_error / count);

        // last extra iter, just compute fitness & mse
        if(iter == criteria.max_iteration_) return result;

        if(std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_) 
        {
            return result;
        }

        for(int i = 0; i < 6; i ++) b_host[i] = Ab_tight[21 + i];//计算外参
////？？？？？

        int shift = 0;
        for(int y = 0; y < 6; y ++) 
        {
            for(int x = y; x < 6; x ++) 
            {
                A_host[x + y * 6] = Ab_tight[shift];
                A_host[y + x * 6] = Ab_tight[shift];
                shift ++;
            }
        }

///算外参
        Mat4x4f extrinsic = eigen_slover_666(A_host.data(), b_host.data());
///变换
        transform_pcd_cuda<<<numBlocks, threadsPerBlock>>>(model_pcd.data(), model_pcd.size(), extrinsic);
        cudaStreamSynchronize(cudaStreamPerThread);

        result.transformation_ = extrinsic * result.transformation_;
    }

    // never arrive here
    return result;
}

template RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f>&, const Scene_nn, const ICPConvergenceCriteria);
device_vector_holder<Vec3f> copy_pcl_host2device(std::vector<Vec3f> &model_pcd)
{
    device_vector_holder<Vec3f> cloud(model_pcd.size());
    thrust::copy(model_pcd.begin(), model_pcd.end(), cloud.begin_thr());
    return cloud;
}


}



