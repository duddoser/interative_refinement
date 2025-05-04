// File: refineSVD8_dynamic.hpp
#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>

using namespace Eigen;

using T = long double;
using Mat = Matrix<T, Dynamic, Dynamic>;

inline void refineSVD8(const Mat& A, Mat& U, Mat& V, Mat& S) {
    int m = A.rows();
    int n = A.cols();
    assert(U.rows() == m && U.cols() == m);
    assert(V.rows() == n && V.cols() == n);

    MatrixXd Ad = A.cast<double>();
    MatrixXd Ud = U.cast<double>();
    MatrixXd Vd = V.cast<double>();

    MatrixXd Im = MatrixXd::Identity(m, m);
    MatrixXd In = MatrixXd::Identity(n, n);

    MatrixXd Ud_1 = Ud.leftCols(n);
    MatrixXd P = Ad * Vd;
    MatrixXd Q = Ad.transpose() * Ud_1;

    VectorXd r(n), t(n);
    MatrixXd Sigma_n = MatrixXd::Zero(n, n);

    for (int i = 0; i < n; i++) {
        double ui_ud = Ud.col(i).transpose() * Ud.col(i);
        double vi_vd = Vd.col(i).transpose() * Vd.col(i);
        r(i) = 1.0 - 0.5 * (ui_ud + vi_vd);
        t(i) = Ud.col(i).transpose() * P.col(i);
        Sigma_n(i, i) = t(i) / (1.0 - r(i));
    }

    S = Mat::Zero(m, n);
    S.block(0, 0, n, n) = Sigma_n.cast<T>();

    MatrixXd P1 = Q - Vd * Sigma_n;
    MatrixXd P2 = P - Ud_1 * Sigma_n;

    MatrixXd P3 = Vd.transpose() * P1;
    MatrixXd P4 = Ud_1.transpose() * P2;

    MatrixXd Q1 = 0.5 * (P3 + P4);
    MatrixXd Q2 = 0.5 * (P3 - P4);

    MatrixXd Ud_2;
    if (m > n) {
        Ud_2 = Ud.rightCols(m - n);
    } else {
        Ud_2 = MatrixXd(m, 0); // совместимо по типу и размеру
    }

    MatrixXd E3, E4, E5;
    if (m > n) {
        MatrixXd Q3 = (1.0 / sqrt(2.0)) * P.transpose() * Ud_2;
        MatrixXd Q4 = (1.0 / sqrt(2.0)) * Ud_2.transpose() * P2;
        E3 = Sigma_n.inverse() * Q3;
        E4 = Q4 * Sigma_n.inverse();
        E5 = 0.5 * (MatrixXd::Identity(m - n, m - n) - Ud_2.transpose() * Ud_2);
    }

    MatrixXd E1 = MatrixXd::Zero(n, n), E2 = MatrixXd::Zero(n, n);
    VectorXd sigma = Sigma_n.diagonal();
    MatrixXd denom1 = sigma.replicate(1, n) - sigma.transpose().replicate(n, 1);
    MatrixXd denom2 = sigma.replicate(1, n) + sigma.transpose().replicate(n, 1);

    E1 = Q1.cwiseQuotient(denom1);
    E2 = Q2.cwiseQuotient(denom2);

    for (int i = 0; i < n; ++i)
        E1(i, i) = 0.5 * r(i);
    E2.diagonal().setZero();

    MatrixXd U1 = Ud_1 + Ud_1 * (E1 - E2);
    if (m > n) U1 += sqrt(2.0) * Ud_2 * E4;

    MatrixXd U_new = MatrixXd::Identity(m, m);
    U_new.leftCols(n) = U1;

    if (m > n) {
        MatrixXd U2 = Ud_2 + Ud_2 * E5 - sqrt(2.0) * Ud_1 * E3;
        U_new.rightCols(m - n) = U2;
    }

    MatrixXd V_new = Vd + Vd * (E1 + E2);

    U = U_new.cast<T>();
    V = V_new.cast<T>();
}
