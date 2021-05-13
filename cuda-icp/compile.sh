#!/bin/sh
mkdir -p build
rm -rf build/* &&\
cd build &&\

cmake .. &&\
make -j8
