#include "svd.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <algorithm>

using namespace std;
using namespace Eigen;
// функция, которая выравнивает U, V; поскольку разложение может иметь неопределённость знаков в матрицах U и V

template<typename MatU, typename MatV>
void alignSVD(MatU &U_computed, MatV &V_computed, const MatU &U_true, const MatV &V_true) {
    int n = U_computed.cols();
    for (int i = 0; i < n; i++) {
        if (U_computed.col(i).dot(U_true.col(i)) < 0) {
            U_computed.col(i) *= -1;
            V_computed.col(i) *= -1;
        }
    }
}

// функция, генерирующая случайное SVD разложение, гарантируется ортогональность U и V

template<typename Scalar, int m, int n>
void generateRandomUSV(Matrix<Scalar, m, m>& U,
                       Matrix<Scalar, m, n>& S,
                       Matrix<Scalar, n, n>& V,
                       pair<Scalar, Scalar> matRange)
{
    Matrix<Scalar, m, m> tempU = Matrix<Scalar, m, m>::Random();
    HouseholderQR<Matrix<Scalar, m, m>> qrU(tempU);
    U = qrU.householderQ();

    Matrix<Scalar, n, n> tempV = Matrix<Scalar, n, n>::Random();
    HouseholderQR<Matrix<Scalar, n, n>> qrV(tempV);
    V = qrV.householderQ();

    S = Matrix<Scalar, m, n>::Zero();
    int k = min(m, n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<Scalar> dist(matRange.first, matRange.second);
    for (int i = 0; i < k; ++i)
        S(i, i) = dist(gen);
}

// запускаем итеративное приближение разложения
// как параметр, можно изменять numTrials, при вызове функции, для многократного повторения теста

template<typename Scalar, int M, int N>
void testSVDRefinementCSV(const string& testName, pair<Scalar, Scalar> matrixInterval, 
                          ofstream &csvFile, int numTrials = 10)
{
    vector<Scalar> noiseLevels = { (Scalar)1e-3, (Scalar)1e-15, (Scalar)1e-18, (Scalar)1e-33 }; // уровни шумов выбраны таким образом, так как это точности long double
    vector<int> iterCounts = { 1, 5, 10, 50 }; // количество итераций нашего алгоритма над каждой матрицей

    for (int trial = 1; trial <= numTrials; trial++) {

        Matrix<Scalar, M, M> U0;
        Matrix<Scalar, M, N> S0;
        Matrix<Scalar, N, N> V0;
        generateRandomUSV<Scalar, M, N>(U0, S0, V0, matrixInterval);
        Matrix<Scalar, M, N> A = U0 * S0 * V0.transpose();

        vector<Scalar> sv;
        int minDim = min(M, N);
        for (int i = 0; i < minDim; i++)
            sv.push_back(S0(i, i));
        sort(sv.begin(), sv.end(), greater<Scalar>());
        long double cond = sv.front() / sv.back();

        for (auto noise : noiseLevels) {
            Matrix<Scalar, M, M> Uc = U0;
            Matrix<Scalar, N, N> Vc = V0;
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<Scalar> noiseDist(0.0, noise);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < M; j++)
                    Uc(i, j) += noiseDist(gen);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    Vc(i, j) += noiseDist(gen);

            for (auto iter : iterCounts) {
                Matrix<Scalar, M, M> Un = Uc;
                Matrix<Scalar, N, N> Vn = Vc;
                Matrix<Scalar, M, N> Sn;
                
                auto t0 = chrono::high_resolution_clock::now();

                for (int k = 1; k < iter; ++k)
                    SVD<Scalar, M, N>::refine(A, Un, Vn, Sn);

                auto t1 = chrono::high_resolution_clock::now();
                double time_ms = chrono::duration<double, milli>(t1 - t0).count();

                alignSVD(Un, Vn, U0, V0);

                // считаем метрики

                long double r1 = (A - Un * Sn * Vn.transpose()).template lpNorm<1>();
                long double r2 = (A - Un * Sn * Vn.transpose()).norm();
                long double Ue = (Un - U0).norm();
                long double Ve = (Vn - V0).norm();
                long double Se = (Sn - S0).norm();

                // записываем в csv файл

                csvFile << M << "x" << N << ",["
                        << matrixInterval.first << ";" << matrixInterval.second << "],"
                        << cond << "," << noise << ","
                        << iter << "," << r1 << ","
                        << r2 << "," << Ue << ","
                        << Se << "," << Ve << ","
                        << time_ms << ","
                        << trial << "\n";
            }
        }
    }
}

int main() {
    ofstream csvFile("svd_results.csv");
    if (!csvFile.is_open()) {
        cerr << "Cannot open file 'svd_results.csv' for writing!" << endl;
        return 1;
    }

    csvFile << "Size,MatrixInterval,Noise,Iter,r1,r2,Ue,SError,Ve,Time_ms,Trials\n";

    int numTrials = 1; // данный параметр можно изменять
    

    // пример вызова функции:

    testSVDRefinementCSV<long double, 4, 3>("Test_4x3", {0, 10}, csvFile, numTrials);
    testSVDRefinementCSV<long double, 4, 3>("Test_4x3", {0, 100}, csvFile, numTrials);
    testSVDRefinementCSV<long double, 5, 5>("Test_5x5", {0, 10}, csvFile, numTrials);
    testSVDRefinementCSV<long double, 5, 5>("Test_5x5", {0, 100}, csvFile, numTrials);
    testSVDRefinementCSV<long double, 6, 4>("Test_6x4", {0, 10}, csvFile, numTrials);
    testSVDRefinementCSV<long double, 6, 4>("Test_6x4", {0, 100}, csvFile, numTrials);
    testSVDRefinementCSV<long double, 52, 50>("Test_52x50", {0, 10}, csvFile, numTrials);
    testSVDRefinementCSV<long double, 52, 50>("Test_52x50", {0, 100}, csvFile, numTrials);


    csvFile.close();
    cout << "CSV file 'svd_results.csv' written successfully." << endl;
    return 0;
}
