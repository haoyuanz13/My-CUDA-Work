#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include "helper.h"
#include "icp_worker.h"


int main(int argc, char const *argv[]) 
{

#ifdef CUDA_ON
    {  // gpu need sometime to warm up
        cudaFree(0);
        // cudaSetDevice(0);

        // cublas also need
        // cublasStatus_t stat;  // CUBLAS functions status
        cublasHandle_t cublas_handle;  // CUBLAS context
        /*stat = */cublasCreate(&cublas_handle);
    }

    // pcl src
    std::string dir_base_src(argv[1]);
    std::cout << "pointcloud src: " << dir_base_src << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model_src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(dir_base_src, *cloud_model_src);
    const int pcl_size_src = cloud_model_src->size();

    // pcl dst
    std::string dir_base_dst(argv[2]);
    std::cout << "pointcloud dst: " << dir_base_dst << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model_dst(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(dir_base_dst, *cloud_model_dst);
    const int pcl_size_dst = cloud_model_dst->size();

    // init cuda icp worker
    float max_dist = 0.1f;      // During ICP, two points will not be considered as a correspondence if they are at least theis far from another
    int max_iter = 10;          // Iterate ICP this many time
    float re_fitness = 1e-5f;   // increase less than this will stop
    float re_rmse = 1e-5f;

    // build icp worker
    static ICPWorker &my_cuda_icp = ICPWorker::GetInstance();
    my_cuda_icp.InitInstance(max_dist, max_iter, re_fitness, re_rmse);

    // do icp multi-times to check memory issue
    helper::Timer timer;
    cv::Mat res_trans_src2dst;    // trans mat [R, t]
    for (int i = 0; i < 10; i ++) 
    {
        std::cout << "[*] pcl src size: " << pcl_size_src << " | pcl dst size: " << pcl_size_dst << std::endl;

        timer.reset();
        
        my_cuda_icp.BuildKdTree(cloud_model_dst);
        res_trans_src2dst = my_cuda_icp.Estimate(cloud_model_src);  // [R, t]

        timer.out("[*] cuda icp");

        std::cout << "\nresult transformation matrix:" << std::endl;
        std::cout << res_trans_src2dst << std::endl;
        std::cout << "\n" << std::endl;
    }

    // write the estimated res into the txt file for vis check
    std::string res_csv_file(argv[3]);
    std::ofstream res_file;
    res_file.open (res_csv_file);
    res_file << "This is the estimated transform matrix [R, t].\n";
    res_file << res_trans_src2dst.at<float>(0, 0) << "," << res_trans_src2dst.at<float>(0, 1) << "," << res_trans_src2dst.at<float>(0, 2) \
        << "," << res_trans_src2dst.at<float>(0, 3) << "\n";
    res_file << res_trans_src2dst.at<float>(1, 0) << "," << res_trans_src2dst.at<float>(1, 1) << "," << res_trans_src2dst.at<float>(1, 2) \
        << "," << res_trans_src2dst.at<float>(1, 3) << "\n";
    res_file << res_trans_src2dst.at<float>(2, 0) << "," << res_trans_src2dst.at<float>(2, 1) << "," << res_trans_src2dst.at<float>(2, 2) \
        << "," << res_trans_src2dst.at<float>(2, 3) << "\n";
    res_file.close();

#endif

    return 0;
}
