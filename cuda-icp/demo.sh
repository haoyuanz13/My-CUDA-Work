EXE_FILE_NAME=demo_icp_worker
DATA_PCL_SRC=`pwd`/data/bun000.ply
DATA_PCL_DST=`pwd`/data/bun001.ply
T_MAT_RES_FILE=`pwd`/res/estimate_res.csv

echo "- exe file name:" ${EXE_FILE_NAME}
echo "- data pcl src dir: " ${DATA_PCL_SRC}
echo "- data pcl dst dir: " ${DATA_PCL_DST}
echo "- res mat csv dir: " ${T_MAT_RES_FILE}

# execution
echo "--> execution cuda icp ..."
./build/${EXE_FILE_NAME} ${DATA_PCL_SRC} ${DATA_PCL_DST} ${T_MAT_RES_FILE}
