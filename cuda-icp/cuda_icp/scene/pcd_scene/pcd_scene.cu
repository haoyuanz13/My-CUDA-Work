#include "pcd_scene.h"


void Scene_nn::copy_kdtree_host2device(KDTree_cpu &cpu_tree, KDTree_cuda &cuda_tree)
{
    cuda_tree.pcd_buffer.__malloc(cpu_tree.pcd_buffer.size());
    thrust::copy(cpu_tree.pcd_buffer.begin(), cpu_tree.pcd_buffer.end(), cuda_tree.pcd_buffer.begin_thr());

    cuda_tree.normal_buffer.__malloc(cpu_tree.normal_buffer.size());
    thrust::copy(cpu_tree.normal_buffer.begin(), cpu_tree.normal_buffer.end(), cuda_tree.normal_buffer.begin_thr());

    cuda_tree.nodes.__malloc(cpu_tree.nodes.size());
    thrust::copy(cpu_tree.nodes.begin(), cpu_tree.nodes.end(), cuda_tree.nodes.begin_thr());
}

void Scene_nn::update_scene_nn_cuda(KDTree_cuda &cuda_tree) 
{
    pcd_ptr = cuda_tree.pcd_buffer.data();
    normal_ptr = cuda_tree.normal_buffer.data();
    node_ptr = cuda_tree.nodes.data();
}