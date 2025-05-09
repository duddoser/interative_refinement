#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include "../refineSVD8_dynamic.hpp"

using namespace std;
using namespace Eigen;

using T = long double;
using Mat = Matrix<T, Dynamic, Dynamic>;

void alignSVD(Mat& Uc, Mat& Vc, const Mat& U0, const Mat& V0) {
    int k = V0.cols();
    for (int i = 0; i < k; ++i) {
        if (Uc.col(i).dot(U0.col(i)) < T(0)) {
            Uc.col(i) *= T(-1);
            Vc.col(i) *= T(-1);
        }
    }
}

int main() {
    vector<pair<int, int>> sizes = {{5,5}, {10,7}, {15,15}, {20,20}, {30,20}, {50,50}, {100,100}};
    vector<pair<double, double>> intervals = {{1, 10}, {1, 100}};
    vector<T> noiseLevels = {1e-15L, 1e-10L, 1e-5L};
    vector<int> iterCounts = {1, 5, 10, 20, 30, 50};

    ofstream out("alg8_test_results.csv");
    out << "Size,Interval,CondNum,Noise,Iter,Rec_l1,Rec_l2,U_err,S_err,V_err,Time_ms\n";

    random_device rd; mt19937 gen(rd());

    for (auto [m, n] : sizes) {
        for (auto [a, b] : intervals) {
            Mat U0 = Mat::Random(m, m), V0 = Mat::Random(n, n);
            HouseholderQR<Mat> qrU(U0), qrV(V0);
            U0 = qrU.householderQ();
            V0 = qrV.householderQ();

            uniform_real_distribution<T> dist(a, b);
            vector<T> sv(n);
            for (int i = 0; i < n; ++i) sv[i] = dist(gen);
            sort(sv.begin(), sv.end(), greater<T>());

            Mat S0 = Mat::Zero(m, n);
            for (int i = 0; i < n; ++i) S0(i, i) = sv[i];

            Mat A = U0 * S0 * V0.transpose();
            long double cond = sv.front() / sv.back();

            for (T noise : noiseLevels) {
                Mat Uc = U0, Vc = V0;
                uniform_real_distribution<T> noiseDist(0.0L, noise);
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < m; ++j)
                        Uc(i, j) += noiseDist(gen);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        Vc(i, j) += noiseDist(gen);

                for (int iter : iterCounts) {
                    Mat Un = Uc, Vn = Vc, Sn;
                    auto t0 = chrono::high_resolution_clock::now();
                    for (int k = 0; k < iter; ++k)
                        refineSVD8(A, Un, Vn, Sn);
                    auto t1 = chrono::high_resolution_clock::now();
                    double time_ms = chrono::duration<double, milli>(t1 - t0).count();

                    alignSVD(Un, Vn, U0, V0);
                    long double r1 = (A - Un * Sn * Vn.transpose()).lpNorm<1>();
                    long double r2 = (A - Un * Sn * Vn.transpose()).norm();
                    long double Ue = (Un - U0).norm();
                    long double Ve = (Vn - V0).norm();
                    long double Se = (Sn - S0).norm();

                    out << m << "x" << n << ",[" << a << ";" << b << "]," << cond << "," << noise << ","
                        << iter << "," << r1 << "," << r2 << "," << Ue << "," << Se << "," << Ve << "," << time_ms << "\n";

                    cout << "Done: " << m << "x" << n << ", noise=" << noise << ", iter=" << iter << endl;
                }
            }
        }
    }

    out.close();
    cout << "All tests complete.\n";
    return 0;
}
