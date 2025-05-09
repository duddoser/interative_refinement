// File: refineSVD8_dynamic.hpp
#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>

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

    MatrixXd Ud_1 = Ud.leftCols(n);

    // Step 1: Compute updated Sigma directly from Uᵗ A V (as in the paper)
    MatrixXd Sigma_n = Ud_1.transpose() * Ad * Vd;

    if (!Sigma_n.allFinite()) {
        std::cerr << "Warning: Invalid Sigma_n\n";
        return;
    }

    S = Mat::Zero(m, n);
    S.block(0, 0, n, n) = Sigma_n.cast<T>();

    MatrixXd P = Ad * Vd;
    MatrixXd Q = Ad.transpose() * Ud_1;

    MatrixXd P1 = Q - Vd * Sigma_n;
    MatrixXd P2 = P - Ud_1 * Sigma_n;

    MatrixXd P3 = Vd.transpose() * P1;
    MatrixXd P4 = Ud_1.transpose() * P2;

    MatrixXd Q1 = 0.5 * (P3 + P4);
    MatrixXd Q2 = 0.5 * (P3 - P4);

    MatrixXd Ud_2 = (m > n) ? Ud.rightCols(m - n) : MatrixXd(m, 0);
    MatrixXd E3, E4, E5;
    if (m > n) {
        MatrixXd Q3 = (1.0 / sqrt(2.0)) * P.transpose() * Ud_2;
        MatrixXd Q4 = (1.0 / sqrt(2.0)) * Ud_2.transpose() * P2;

        if (Sigma_n.determinant() < 1e-14) {
            std::cerr << "Warning: Sigma_n nearly singular during E3/E4\n";
            return;
        }

        E3 = Sigma_n.inverse() * Q3;
        E4 = Q4 * Sigma_n.inverse();
        E5 = 0.5 * (MatrixXd::Identity(m - n, m - n) - Ud_2.transpose() * Ud_2);
    }

    MatrixXd E1 = MatrixXd::Zero(n, n), E2 = MatrixXd::Zero(n, n);
    VectorXd sigma = Sigma_n.diagonal();
    MatrixXd denom1 = sigma.replicate(1, n) - sigma.transpose().replicate(n, 1);
    MatrixXd denom2 = sigma.replicate(1, n) + sigma.transpose().replicate(n, 1);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double d1 = std::abs(denom1(i, j)) < 1e-14 ? 1e-14 : denom1(i, j);
            double d2 = std::abs(denom2(i, j)) < 1e-14 ? 1e-14 : denom2(i, j);
            E1(i, j) = Q1(i, j) / d1;
            E2(i, j) = Q2(i, j) / d2;
        }
    }

    E1.diagonal() = 0.5 * VectorXd::Ones(n);
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

    // Re-orthogonalize to prevent error accumulation
    HouseholderQR<MatrixXd> qrU(U_new);
    U_new = qrU.householderQ();
    HouseholderQR<MatrixXd> qrV(V_new);
    V_new = qrV.householderQ();

    if (!U_new.allFinite() || !V_new.allFinite()) {
        std::cerr << "Warning: U or V contains NaNs or Infs after update\n";
        return;
    }

    U = U_new.cast<T>();
    V = V_new.cast<T>();
}
