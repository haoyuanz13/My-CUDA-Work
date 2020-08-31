#!/bin/sh
mkdir -p `pwd`/build

rm -rf build/* &&\
cd build &&\

cmake .. -DCUDNN_LIBRARY=/data/cuda/cuda-10.0/cudnn/v7.6.5/lib64/libcudnn.so &&\
make -j8