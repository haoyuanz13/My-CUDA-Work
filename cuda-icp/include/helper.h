#pragma once

#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "icp.h"

namespace helper 
{

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    
    double elapsed() const 
    {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); 
    }
    
    void out(std::string message = "") 
    {
        double t = elapsed();
        std::cout << message << " elasped time:" << t * 1000 << " ms" << std::endl;
        reset();
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;

};

cv::Rect get_bbox(cv::Mat depth);
cv::Mat mat4x4f2cv(Mat4x4f& mat4);
cv::Mat view_dep(cv::Mat dep);
bool isRotationMatrix(cv::Mat &R);
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat R);
cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f theta);

}  // namespace helper

