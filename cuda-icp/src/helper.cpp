#include "helper.h"

namespace helper 
{

cv::Rect get_bbox(cv::Mat depth) 
{
    cv::Mat mask = depth > 0;
    cv::Mat Points;
    cv::findNonZero(mask, Points);
    return cv::boundingRect(Points);
}

cv::Mat mat4x4f2cv(Mat4x4f& mat4) 
{
    cv::Mat mat_cv(4, 4, CV_32F);
    mat_cv.at<float>(0, 0) = mat4[0][0]; mat_cv.at<float>(0, 1) = mat4[0][1];
    mat_cv.at<float>(0, 2) = mat4[0][2]; mat_cv.at<float>(0, 3) = mat4[0][3];

    mat_cv.at<float>(1, 0) = mat4[1][0]; mat_cv.at<float>(1, 1) = mat4[1][1];
    mat_cv.at<float>(1, 2) = mat4[1][2]; mat_cv.at<float>(1, 3) = mat4[1][3];

    mat_cv.at<float>(2, 0) = mat4[2][0]; mat_cv.at<float>(2, 1) = mat4[2][1];
    mat_cv.at<float>(2, 2) = mat4[2][2]; mat_cv.at<float>(2, 3) = mat4[2][3];

    mat_cv.at<float>(3, 0) = mat4[3][0]; mat_cv.at<float>(3, 1) = mat4[3][1];
    mat_cv.at<float>(3, 2) = mat4[3][2]; mat_cv.at<float>(3, 3) = mat4[3][3];

    return mat_cv;
}

cv::Mat view_dep(cv::Mat dep) 
{
    cv::Mat map = dep;
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    map.convertTo(adjMap, CV_8UC1, 255 / (max - min), -min);
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
    
    return falseColorsMap;
}

bool isRotationMatrix(cv::Mat &R) 
{
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
    
    return norm(I, shouldBeIdentity) < 1e-5;
}

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat R) 
{
    assert(isRotationMatrix(R));
    float sy = std::sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));

    bool singular = sy < 1e-6f; // If

    float x, y, z;
    if (!singular)
    {
        x = std::atan2(R.at<float>(2, 1) , R.at<float>(2, 2));
        y = std::atan2(-R.at<float>(2, 0), sy);
        z = std::atan2(R.at<float>(1, 0), R.at<float>(0, 0));
    }

    else
    {
        x = std::atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
        y = std::atan2(-R.at<float>(2, 0), sy);
        z = 0;
    }

    return cv::Vec3f(x, y, z);
}

cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<float>(3, 3) <<
               1,       0,              0,
               0,       std::cos(theta[0]),   -std::sin(theta[0]),
               0,       std::sin(theta[0]),   std::cos(theta[0])
               );
    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<float>(3, 3) <<
               std::cos(theta[1]),    0,      std::sin(theta[1]),
               0,               1,      0,
               -std::sin(theta[1]),   0,      std::cos(theta[1])
               );
    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<float>(3, 3) <<
               std::cos(theta[2]),    -std::sin(theta[2]),      0,
               std::sin(theta[2]),    std::cos(theta[2]),       0,
               0,               0,                  1);
    
    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;
    
    return R;
}

}  // namespace helper
