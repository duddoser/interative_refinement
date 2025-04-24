#include "alg8.hpp"
#include <iostream>
#include <eigen3/Eigen/Dense>

int main() {
    using namespace Eigen;
    using namespace std;

    cout.precision(4);
    cout << fixed;

    // Случайная матрица 7x5
    Eigen::Matrix<float, Dynamic, Dynamic> A(10,9);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 68, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80, 81,
        3, 9, (float)4.98942, (float)0.324235,  443534, 345, (float)56.543853, (float)450.435234, (float)43.34353221;

    // Начальное SVD (для сравнения)
    BDCSVD<MatrixXf> svd(A, ComputeFullU | ComputeFullV);

    cout << "A:\n" << A << endl << endl;
    cout << "U:\n" << svd.matrixU() << endl;
    cout << "Singular Values:\n" << svd.singularValues() << endl;
    cout << "V:\n" << svd.matrixV() << endl << endl;

    //Создание сигма-матрицы для проверки реконструкции
    VectorXf singular_values = svd.singularValues();
    MatrixXf Sigma = MatrixXf::Zero(A.rows(), A.cols());
    Sigma.diagonal() = singular_values;

    cout << "Reconstructed A (U * S * V^T):\n" 
         << svd.matrixU() * Sigma * svd.matrixV().transpose() << endl << endl;

    cout << "=== Refinement by Algorithm 8 ===\n\n";

    // Уточнение через Algorithm 8
    Ogita_Aishima_SVD<float, 10, 9> Ans(A);

    cout << "U (refined):\n" << Ans.matrixU() << endl;
    cout << "Sigma (refined):\n" << Ans.singularValues() << endl;
    cout << "V (refind):\n" << Ans.matrixV() << endl;

    MatrixXf A_refined = Ans.matrixU() * Ans.singularValues() * Ans.matrixV().transpose();

    cout << "\nReconstructed A after refinement:\n" << A_refined << endl;

    // Нормы для оценки качества
    cout << "\n||A - A_refined||_F = " << (A - A_refined).norm() << endl;

    return 0;
}
