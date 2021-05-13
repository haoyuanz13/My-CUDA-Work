#include "icp.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>



namespace cuda_icp{
//EIGEN_MAKE_ALIGNED_OPERATOR_NEW


Mat4x4f eigen_to_custom(const Eigen::Matrix4f& extrinsic) 
{
    Mat4x4f result;
    for(uint32_t i=0; i<4; i++){
        for(uint32_t j=0; j<4; j++){
            result[i][j] = extrinsic(i, j);
        }
    }
    return result;
}

Mat4x4f eigen_slover_666(float *A, float *b)
{
    //Eigen::Matrix<double, 6, 6> Aa;

    Eigen::Matrix<float, 6, 6> A_eigen(A);

    Eigen::Matrix<float, 6, 1> b_eigen(b);
//    const Eigen::Matrix<double, 6, 1> update = A_eigen.cast<double>().ldlt().solve(b_eigen.cast<double>());
    const Eigen::Matrix<float, 6, 1> update = A_eigen.ldlt().solve(b_eigen);
    Eigen::Matrix<float, 4,4> extrinsic;

    extrinsic.setIdentity();
    extrinsic.block<3, 3>(0, 0) =
            (Eigen::AngleAxisf(update(2), Eigen::Vector3f::UnitZ()) *
             Eigen::AngleAxisf(update(1), Eigen::Vector3f::UnitY()) *
             Eigen::AngleAxisf(update(0), Eigen::Vector3f::UnitX())).matrix();
    extrinsic.block<3, 1>(0, 3) = update.block<3, 1>(3, 0);


    return eigen_to_custom(extrinsic.cast<float>());

}


}


