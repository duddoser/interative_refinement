// File: test_alg8_special.cpp
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <random>
#include <iomanip>
#include "../refineSVD8_dynamic.hpp"

using namespace std;
using namespace Eigen;

using T = long double;
using Mat = Matrix<T, Dynamic, Dynamic>;

void printMatrix(ofstream& out, const string& name, const Mat& M) {
    out << name << " (" << M.rows() << "x" << M.cols() << "):\n";
    out << M << "\n\n";
}

int main() {
    ofstream out("alg8_special_output.txt");
    if (!out.is_open()) {
        cerr << "Failed to open output file.\n";
        return 1;
    }

    const int m = 5;
    const int n = 3;

    Mat U0 = Mat::Random(m, m);
    Mat V0 = Mat::Random(n, n);
    HouseholderQR<Mat> qrU(U0), qrV(V0);
    U0 = qrU.householderQ();
    V0 = qrV.householderQ();

    vector<T> sv = {10.0, 5.0, 1.0};
    Mat S0 = Mat::Zero(m, n);
    for (int i = 0; i < n; ++i)
        S0(i, i) = sv[i];

    Mat A = U0 * S0 * V0.transpose();

    Mat Uc = U0, Vc = V0;
    T noise = 1e-4;
    mt19937 gen(42);
    uniform_real_distribution<T> dist(0.0, noise);
    for (int i = 0; i < Uc.rows(); ++i)
        for (int j = 0; j < Uc.cols(); ++j)
            Uc(i, j) += dist(gen);
    for (int i = 0; i < Vc.rows(); ++i)
        for (int j = 0; j < Vc.cols(); ++j)
            Vc(i, j) += dist(gen);

    Mat Sn;
    refineSVD8(A, Uc, Vc, Sn);

    out << fixed << setprecision(6);
    printMatrix(out, "A", A);
    printMatrix(out, "U0", U0);
    printMatrix(out, "S0", S0);
    printMatrix(out, "V0", V0);
    printMatrix(out, "Uc (after refinement)", Uc);
    printMatrix(out, "Vn (after refinement)", Vc);
    printMatrix(out, "Sn (after refinement)", Sn);

    Mat A_reconstructed = Uc * Sn * Vc.transpose();
    T resNorm = (A - A_reconstructed).norm();
    out << "||A - U*S*V^T|| = " << resNorm << "\n";

    out.close();
    cout << "Output written to alg8_special_output.txt\n";
    return 0;
}
