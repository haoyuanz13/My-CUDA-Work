#include "icp_worker.h"


#ifdef CUDA_ON


ICPWorker &ICPWorker::GetInstance() 
{
    static ICPWorker instance;
    return instance;
}

void ICPWorker::InitInstance(float max_dist, int max_iter, float re_fitness, float re_rmse) 
{
    // reset timer count
    timer_.reset();

    // assign icp vars
    max_dist_ = max_dist;

    // init pcl involved vars
    pcl_normals_ = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal>);
    pcl_kdtree_ = pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>());
    normals_est_.setKSearch(20);
    normals_est_.setSearchMethod(pcl_kdtree_);

    // init icp criteria
    icpcc_.max_iteration_ = max_iter;
    icpcc_.relative_fitness_ = re_fitness;
    icpcc_.relative_rmse_ = re_rmse;

    // init scene instance
    scene_.init_scene_nn_cpu(kdtree_cpu_, max_dist_);
    // scene_.init_scene_nn_cuda(kdtree_cuda_);

    // show time cost
    timer_.out("--> [init icp_worker]");
}

void ICPWorker::BuildKdTree(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_src_ptr) 
{
    // reset timer count
    timer_.reset();

    // compute pcl normals
    normals_est_.setInputCloud(pcl_src_ptr);
    normals_est_.compute(*pcl_normals_);

    // build kd tree cpu
    for(int i = 0; i < pcl_src_ptr->points.size(); i ++)
    {
        float x = pcl_src_ptr->points[i].x;
        float y = pcl_src_ptr->points[i].y;
        float z = pcl_src_ptr->points[i].z;
        kdtree_cpu_.pcd_buffer.push_back({x, y, z});

        float nx = pcl_normals_->points[i].normal_x;
        float ny = pcl_normals_->points[i].normal_y;
        float nz = pcl_normals_->points[i].normal_z;
        kdtree_cpu_.normal_buffer.push_back({nx, ny, nz});
    }

    kdtree_cpu_.build_tree();

    // update scene cuda instance
    scene_.copy_kdtree_host2device(kdtree_cpu_, kdtree_cuda_);
    scene_.update_scene_nn_cuda(kdtree_cuda_);

    // show time cost
    timer_.out("--> [build kdtree]");
}

cv::Mat ICPWorker::Estimate(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_dst_ptr) 
{
    // reset timer count
    timer_.reset();

    // get src pcl data 
    std::vector<::Vec3f> pcd_dst_host;
    for(int i = 0; i < pcl_dst_ptr->points.size(); i ++)
    {
        float x = pcl_dst_ptr->points[i].x;
        float y = pcl_dst_ptr->points[i].y;
        float z = pcl_dst_ptr->points[i].z;
        pcd_dst_host.push_back({x, y, z});

    }
    auto pcd_dst_device = cuda_icp::copy_pcl_host2device(pcd_dst_host);

    // icp exe
    auto result_cuda = cuda_icp::ICP_Point2Plane_cuda(pcd_dst_device, scene_, icpcc_);

    cv::Mat result_T = helper::mat4x4f2cv(result_cuda.transformation_);

    // reset cpu_tree and cuda_tree buffer
    kdtree_cuda_.pcd_buffer.__free();
    kdtree_cuda_.normal_buffer.__free();
    kdtree_cuda_.nodes.__free();

    kdtree_cpu_.pcd_buffer.clear();
    kdtree_cpu_.normal_buffer.clear();
    kdtree_cpu_.nodes.clear();

    // show time cost
    timer_.out("--> [icp estimation]");

    return result_T;
}

#endif
