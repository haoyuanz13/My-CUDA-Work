#!/bin/sh
mkdir -p `pwd`/build

rm -rf build/* &&\
cd build &&\

cmake .. &&\
make -j8