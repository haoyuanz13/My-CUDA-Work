DATA_PCL_SRC=`pwd`/data/bun000.ply
DATA_PCL_DST=`pwd`/data/bun001.ply
T_MAT_RES_FILE=`pwd`/res/estimate_res.csv

echo "- data pcl src dir: " ${DATA_PCL_SRC}
echo "- data pcl dst dir: " ${DATA_PCL_DST}
echo "- res mat csv dir: " ${T_MAT_RES_FILE}

# vis res
python `pwd`/scripts/vis_res.py \
    --src_pcl_dir ${DATA_PCL_SRC} \
    --dst_pcl_dir ${DATA_PCL_DST} \
    --res_csv_dir ${T_MAT_RES_FILE}
