import argparse
import numpy as np
import open3d as o3d
import csv
import pdb


'''
    define arguments
'''
def parse_args():
    parser = argparse.ArgumentParser("estimated trans vis")
    parser.add_argument('--src_pcl_dir', type=str, help='the src pcl data file')
    parser.add_argument('--dst_pcl_dir', type=str, help='the dst pcl data file')
    parser.add_argument('--res_csv_dir', type=str, help='the estimated transform mat')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get args
    args = parse_args()

    # src pcl. vis as red
    src_pcl_data = o3d.io.read_point_cloud(args.src_pcl_dir)
    src_pcl_num = len(src_pcl_data.points)
    print ( '\nload src from {} succeed!'.format(args.src_pcl_dir) )
    print (src_pcl_data)
    print (o3d.np.asarray(src_pcl_data.points))
    src_pcl_color = np.zeros([src_pcl_num, 3])
    src_pcl_color[:, 0] = 1  # red color for src pcl, the order is rgb
    src_pcl_data.colors = o3d.utility.Vector3dVector(src_pcl_color)

    # dst pcl, vis as green
    dst_pcl_data = o3d.io.read_point_cloud(args.dst_pcl_dir)
    dts_pcl_num = len(dst_pcl_data.points)
    print ( 'load dst from {} succeed!'.format(args.dst_pcl_dir) )
    print (dst_pcl_data)
    print (o3d.np.asarray(dst_pcl_data.points))
    dst_pcl_color = np.zeros([dts_pcl_num, 3])
    dst_pcl_color[:, 1] = 1  # green color for dst pcl, the order is rgb
    dst_pcl_data.colors = o3d.utility.Vector3dVector(dst_pcl_color)
    
    # vis raw res
    o3d.visualization.draw_geometries(
        geometry_list=[dst_pcl_data, src_pcl_data], 
        window_name='raw pcl', 
        width=1280, 
        height=720
    )
    
    # trans src pcl and compare
    print ('\nTransform src pcl using the estimated matrix from cuda icp ...')
    
    # load estimated res mat [R, t] and set
    trans_mat_src2dst = np.zeros([4, 4])
    trans_mat_src2dst[3, 3] = 1
    with open(args.res_csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_cout = 0
        for row in csv_reader:
            if line_cout > 0:
                trans_mat_src2dst[line_cout - 1, 0] = float(row[0])
                trans_mat_src2dst[line_cout - 1, 1] = float(row[1])
                trans_mat_src2dst[line_cout - 1, 2] = float(row[2])
                trans_mat_src2dst[line_cout - 1, 3] = float(row[3])

            line_cout += 1
    
    print ('The estimated transform mat from src to dst pcl is: ')
    print (trans_mat_src2dst)

    # trans src pcl
    src_pcl_pts_np = np.ones([4, src_pcl_num])
    src_pcl_pts_np[0:-1, :] = o3d.np.asarray(src_pcl_data.points).T
    src_pcl_pts_np_trans = np.matmul(trans_mat_src2dst, src_pcl_pts_np).T
    src_pcl_data.points = o3d.utility.Vector3dVector(src_pcl_pts_np_trans[:, 0:-1])
    print (src_pcl_data)
    print (o3d.np.asarray(src_pcl_data.points))

    # vis the estimate res
    o3d.visualization.draw_geometries(
        geometry_list=[dst_pcl_data, src_pcl_data], 
        window_name='estimated res', 
        width=1280, 
        height=720
    )

    # done the vis
    print ('\nThe vis task completed!')