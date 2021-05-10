#ifndef _MORPHOLOGY_H_
#define _MORPHOLOGY_H_
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
using namespace cv;


/*
    erode image
*/
void ErodeTwoStepShared(void *src, void *dst, 
    int radio, int width, int height, cudaStream_t& stream);

/*
    erode image updated version
    - extract cudaMalloc out of the callback func
*/
void ErodeTwoStepShared(void *src, void *temp, void *dst, 
    int radio, int width, int height, cudaStream_t& stream);

/*
    erode image updated version
    - extract cudaMalloc out of the callback func
    - introduce batchsize, but has possible logic problem
*/
void ErodeTwoStepShared(const int batch_size, void *src, void *temp, void *dst, 
    int radio, int width, int height, cudaStream_t& stream);

/*
    dilate image
*/
void DilateTwoStepShared(void *src, void *dst, 
    int radio, int width, int height, cudaStream_t& stream);

/*
    dilate image updated version, extract cudaMalloc out of the callback func
*/
void DilateTwoStepShared(void *src, void *temp, void *dst, 
    int radio, int width, int height, cudaStream_t& stream);

#endif
