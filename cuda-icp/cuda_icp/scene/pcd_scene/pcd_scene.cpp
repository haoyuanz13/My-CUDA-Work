#include "pcd_scene.h"
#include <numeric>


void Scene_nn::init_scene_nn_cpu(KDTree_cpu &cpu_tree, float max_dist)
{
    max_dist_diff = max_dist;
    pcd_ptr = cpu_tree.pcd_buffer.data();
    normal_ptr = cpu_tree.normal_buffer.data();
    node_ptr = cpu_tree.nodes.data();
}

void KDTree_cpu::build_tree(int max_num_pcd_in_leaf)
{
    assert(pcd_buffer.size() > 0 && pcd_buffer.size() == normal_buffer.size()
           && "no pcd yet, or pcd size != normal size");

    // build index vector for 1v1 mapping
    std::vector<int> index;
    index.resize(pcd_buffer.size());
    std::iota (std::begin(index), std::end(index), 0);  // Fill with 0, 1, 2, ...

    std::vector<int> index_buffer(index.size());

    // root
    nodes.resize(1);
    nodes[0].left = 0;
    nodes[0].right = index.size();

    bool exist_new_nodes = false;
    size_t num_nodes_now = 1;
    size_t num_nodes_last_last_turn = 0;
    size_t num_nodes_now_last_turn = 0;
    bool stop = false;

    while (!stop) 
    { // when we have new nodes to split, go

        nodes.resize(num_nodes_now * 2 + 1); // we may increase now + 1 in 1 turn

        exist_new_nodes = false; // reset
        num_nodes_last_last_turn = num_nodes_now_last_turn;
        num_nodes_now_last_turn = num_nodes_now; // for iter, avoid reaching new node in 1 turn

        // if you want to implement cuda version, paralleling this loop looks fine
        for(size_t node_iter = num_nodes_last_last_turn; node_iter < num_nodes_now_last_turn; node_iter ++) 
        {

            // not a leaf
            if(nodes[node_iter].right - nodes[node_iter].left > max_num_pcd_in_leaf) 
            {
                // split start <----------------------
                // get bbox
                float x_min = FLT_MAX; float x_max = -FLT_MAX;  // fxxk, FLT_MIN is 0
                float y_min = FLT_MAX; float y_max = -FLT_MAX;
                float z_min = FLT_MAX; float z_max = -FLT_MAX;
                for(int idx_iter = nodes[node_iter].left; idx_iter < nodes[node_iter].right; idx_iter ++) 
                {
                    const auto& p = pcd_buffer[index[idx_iter]];
                    if(p.x > x_max) x_max = p.x;
                    if(p.x < x_min) x_min = p.x;
                    if(p.y > y_max) y_max = p.y;
                    if(p.y < y_min) y_min = p.y;
                    if(p.z > z_max) z_max = p.z;
                    if(p.z < z_min) z_min = p.z;
                }

                // select split dim & value
                int split_dim = 0;
                float split_val = 0;
                float span_xyz[3], split_v_xyz[3];
                float max_span = -FLT_MAX;
                span_xyz[0] = x_max - x_min; split_v_xyz[0] = (x_min + x_max) / 2;
                span_xyz[1] = y_max - y_min; split_v_xyz[1] = (y_min + y_max) / 2;
                span_xyz[2] = z_max - z_min; split_v_xyz[2] = (z_min + z_max) / 2;
                for(int span_iter = 0; span_iter < 3; span_iter ++) 
                {
                    if(span_xyz[span_iter] > max_span) 
                    {
                        max_span = span_xyz[span_iter];
                        split_dim = span_iter;
                        split_val = split_v_xyz[span_iter];
                    }
                }

                // reorder index
                int left_iter = nodes[node_iter].left;
                int right_iter = nodes[node_iter].right - 1;
                float split_low = -FLT_MAX;
                float split_high = FLT_MAX;

                bool lr_switch = true;

                for(int idx_iter = nodes[node_iter].left; idx_iter < nodes[node_iter].right; idx_iter ++) 
                {
                    float p = pcd_buffer[index[idx_iter]][split_dim];
                    //if(p < split_val){
                    if(p == split_val) lr_switch = !lr_switch;

                    if( p < split_val || (p == split_val && lr_switch) ) 
                    {
                        index_buffer[left_iter] = index[idx_iter];
                        left_iter ++;
                        if(p > split_low) split_low = p;
                    }
                    
                    else 
                    {
                        index_buffer[right_iter] = index[idx_iter];
                        right_iter --;
                        if(p < split_high) split_high = p;
                    }
                }

                assert(left_iter == right_iter + 1 && "left & right should meet");
                split_val = (split_low + split_high) / 2;  // reset split_val to middle

                for(int idx_iter = nodes[node_iter].left; idx_iter < nodes[node_iter].right; idx_iter ++) 
                {
                    index[idx_iter] = index_buffer[idx_iter];
                }
                // split success <----------------------


                // update parent
                nodes[node_iter].child1 = num_nodes_now;
                nodes[node_iter].child2 = num_nodes_now + 1;
                nodes[node_iter].split_v = split_val;
                nodes[node_iter].split_dim = split_dim;
                nodes[node_iter].bbox[0] = x_min;  nodes[node_iter].bbox[1] = x_max;
                nodes[node_iter].bbox[2] = y_min;  nodes[node_iter].bbox[3] = y_max;
                nodes[node_iter].bbox[4] = z_min;  nodes[node_iter].bbox[5] = z_max;

                // update child
                nodes[num_nodes_now].left = nodes[node_iter].left;
                nodes[num_nodes_now].right = left_iter;
                nodes[num_nodes_now].parent = node_iter;

                nodes[num_nodes_now + 1].left = left_iter;
                nodes[num_nodes_now + 1].right = nodes[node_iter].right;
                nodes[num_nodes_now + 1].parent = node_iter;

                num_nodes_now += 2;
                if(!exist_new_nodes) exist_new_nodes = true;
            }
        }

        if(!exist_new_nodes) stop = true;
    }

    // we may give nodes more memory while spliting
    nodes.resize(num_nodes_now);

    // reorder pcd normal according to index, so avoid using index when query
    std::vector<Vec3f> v3f_buffer(pcd_buffer.size());
    for(size_t i = 0; i < pcd_buffer.size(); i ++) 
    {
        v3f_buffer[i] = pcd_buffer[index[i]];
    }
    pcd_buffer = v3f_buffer;

    for(size_t i = 0; i < normal_buffer.size(); i ++) 
    {
        v3f_buffer[i] = normal_buffer[index[i]];
    }
    normal_buffer = v3f_buffer;
}
