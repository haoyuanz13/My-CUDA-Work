# cuda gray normalize demo
EXE_FILE_NAME=my_cuda_gray_norm
# EXE_FILE_NAME=my_cuda_bgr2rgb_norm
# EXE_FILE_NAME=my_cuda_center_aligned_padding
# EXE_FILE_NAME=my_cuda_bilinear_inter_resize
echo "- exe file name:" ${EXE_FILE_NAME}

# execution
echo "--> kernel execution ..."
./build/${EXE_FILE_NAME}