# cuda gray normalize demo
EXE_FILE_NAME=my_cuda_gray_norm
# EXE_FILE_NAME=my_cuda_bgr2rgb_norm
echo "- exe file name:" ${EXE_FILE_NAME}

# execution
echo "--> kernel execution ..."
./build/${EXE_FILE_NAME}