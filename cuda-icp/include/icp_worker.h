# pragma once

#include <Eigen/Dense>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include "helper.h"

#ifdef CUDA_ON

class ICPWorker 
{
public:
    /**
     * @brief Copy constructor is forbidden in singleton pattern.
     */
    ICPWorker(ICPWorker const &) = delete;

    /**
     * @brief Copy assignment operator is forbidden in singleton pattern.
     */
    void operator=(ICPWorker const &) = delete;

    /**
     * @brief Get the unique instance of this class.
     *
     * The unique instance is loaded lazily.
     * @return Reference to the unique instance.
     */
    static ICPWorker &GetInstance();

    /**
     * @brief initialize the icp worker instance.
     * 
     * @param max_dist the max distance to match neighbor point
     * @param max_iter the max iteration times for the icp
     * @param re_fitness the fit threshold
     * @param re_rmse the mse threshold
     */
    void InitInstance(float max_dist=2.0f, int max_iter=30, 
        float re_fitness=1e-5f, float re_rmse=1e-5f);

    /**
     * @brief build the kdtree based on the src pointcloud.
     * 
     * @param pcl_src_ptr the pointer to the source pointcloud
     */
    void BuildKdTree(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_src_ptr);

    /**
     * @brief implement the icp and estimate the transformation between src and dst clouds.
     * 
     * @param pcl_dst_ptr the pointer to the dst pointcloud
     * @return the estimated transformation
     */
    cv::Mat Estimate(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_dst_ptr);

private:
    /**
     * @brief default constructor
     */
    ICPWorker() 
    { 
        std::cout << "[*] A new cuda icp worker is created !" << std::endl; 
    };

    /**
     * @brief default destroy
     */
    ~ICPWorker() {};

    // icp involved vars
    float max_dist_;

    // the nearest neighbor involved vars
    Scene_nn scene_;
    KDTree_cpu kdtree_cpu_;
    KDTree_cuda kdtree_cuda_;

    // pcl involved vars
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normals_est_;
    pcl::PointCloud<pcl::Normal>::Ptr pcl_normals_;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr pcl_kdtree_;

    // icp criteria
    cuda_icp::ICPConvergenceCriteria icpcc_;

    // timer count
    helper::Timer timer_;
};

#endif