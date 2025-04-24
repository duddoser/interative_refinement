#include <complex>
#include <iostream>
#include "svd_complex.h"
#include <Eigen/Dense>

int main()
{
    using namespace std;
    using namespace Eigen;
    Eigen::Matrix<complex<float>, Dynamic, Dynamic> A(5,4);

    A << complex<float>(float(1.0),  float(2.0)),   complex<float>(float(3.0),   float(-1.0)),  complex<float>(float(-4.5),  float(0.5)),   complex<float>(float(2.2),  float(-3.3)),
         complex<float>(float(0.0),  float(-1.0)),  complex<float>(float(5.5),   float(2.2)),   complex<float>(float(3.3),   float(3.3)),   complex<float>(float(-6.7), float(4.4)),
         complex<float>(float(2.1),  float(1.1)),   complex<float>(float(-3.4),  float(-2.5)),  complex<float>(float(4.2),   float(0.0)),   complex<float>(float(1.1),  float(-1.1)),
         complex<float>(float(7.8),  float(-3.3)),  complex<float>(float(-4.2),  float(4.1)),   complex<float>(float(0.0),   float(0.0)),   complex<float>(float(5.3),  float(-2.2)),
         complex<float>(float(-6.5), float(1.3)),   complex<float>(float(8.1),   float(-6.7)),  complex<float>(float(2.3),   float(-3.4)),  complex<float>(float(-4.6), float(0.7));

    Eigen::JacobiSVD<Eigen::MatrixXcf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    SVD<std::complex<float>, 5, 4> Ans(A);

    cout << Ans.matrixU() * Ans.singularValues() * Ans.matrixV().adjoint() << "\n" << "\n";
    cout << Ans.matrixU() << "\n";
    cout << Ans.matrixV() << "\n";
    cout << Ans.singularValues() << "\n" << "\n";

    // Восстанавливаем матрицу через Eigen::JacobiSVD для сравнения
    // Singular values для комплексных SVD возвращаются в виде вектора вещественных чисел
    Array<float,1, Dynamic> sigm = svd.singularValues();
    Eigen::MatrixXcf I(5, 4);
    I.setZero();
    I.block(0,0,4,4) = sigm.matrix().asDiagonal();
    Eigen::MatrixXcf U = svd.matrixU();
    //cout << U * I * svd.matrixV().adjoint() << "\n" << "\n";

    return 0;
}

// g++ -Wa,-mbig-obj -O2 -IC:/cpp/eigen-3.4.0 svd_test.cpp -o svd_test.exe
// svd_test.exe
