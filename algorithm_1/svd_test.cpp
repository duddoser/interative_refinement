#define EIGEN_UNROLLING_LIMIT 0

#include <iostream>
#include "svd.h"
#include <Eigen/Dense>

int main()
{
    using namespace std;
    using namespace Eigen;
    
    Eigen::Matrix<float, 95, 94> A = Matrix<float, 95, 94>::Random();
    //cout << "Input matrix A:\n\n" << A << "\n\n";
    Eigen::BDCSVD<Eigen::MatrixXf> svd(A, ComputeFullU | ComputeFullV);
    
    SVD<float, 95, 94> Ans(A);
    
    Eigen::Matrix<float, Dynamic, Dynamic> A_reconstructed = Ans.matrixU() * Ans.singularValues() * Ans.matrixV().transpose();
    //cout << "A_reconstructed:\n\n" <<  A_reconstructed << "\n\n";
    float reconstruction_error = (A - A_reconstructed).norm() / A.norm();
    cout << "Reconstruction error (my SVD, Frobenius norm): " 
         << reconstruction_error << "  \n\n";
         
    Array<float, 1, Dynamic> sigm(94);
    sigm = svd.singularValues();
    Eigen::Matrix<float, Dynamic, Dynamic> I(95, 94);
    I.setZero();
    I.block(0, 0, 94, 94) = sigm.matrix().asDiagonal();
    Eigen::Matrix<float, Dynamic, Dynamic> U = svd.matrixU();
    Eigen::Matrix<float, Dynamic, Dynamic> A_eigen = U * I * svd.matrixV().transpose();
    //cout << "Eigen A matrix:\n" << A_eigen << "\n\n";
    float reconstruction_error_eigen = (A - A_eigen).norm() / A.norm();
    cout << "Reconstruction error (BDCSVD, Frobenius norm): " 
         << reconstruction_error_eigen << "\n\n";
         
    return 0;
}

// Пример компиляции (Windows):
// g++ -Wa,-mbig-obj -O2 -IC:/cpp/eigen-3.4.0 svd_test.cpp -o svd_test.exe
// svd_test.exe
