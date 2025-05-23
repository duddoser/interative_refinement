// File: refineSVD8_dynamic.hpp
#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace Eigen;

using T = long double;
T sqrt2 = std::sqrtl(T(2));
using Mat = Matrix<T, Dynamic, Dynamic>;

inline void refineSVD8(const Mat& A, Mat& U, Mat& V, Mat& S) {
    int m = A.rows();
    int n = A.cols();
    assert(U.rows() == m && U.cols() == m);
    assert(V.rows() == n && V.cols() == n);

    Mat Ad = A.cast<T>();
    Mat Ud = U.cast<T>();
    Mat Vd = V.cast<T>();

    Mat Ud_1 = Ud.leftCols(n);

    Mat P = Ad * Vd;

    Matrix<T, Dynamic, Dynamic> Sigma_n = Matrix<T, Dynamic, Dynamic>::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        T ui_norm_sq = Ud_1.col(i).squaredNorm();   // ||uᵢ||²
        T vi_norm_sq = V.col(i).squaredNorm();      // ||vᵢ||²
        T rii = (1 - (ui_norm_sq + vi_norm_sq) / T(2));
                   // отдельно вычисляем A * vᵢ
        T tii = Ud_1.col(i).dot(P.col(i));              

        if (std::abs(rii) < 1e-14) {
            Sigma_n(i, i) = tii;
        } else {
            Sigma_n(i, i) = tii / rii;
        }
    }


        if (!Sigma_n.allFinite()) {
            std::cerr << "Warning: Invalid Sigma_n\n";
            return;
        }

    S = Mat::Zero(m, n);
    S.block(0, 0, n, n) = Sigma_n.cast<T>();

    Mat Q = Ad.transpose() * Ud_1;

    Mat P1 = Q - Vd * Sigma_n;
    Mat P2 = P - Ud_1 * Sigma_n;

    Mat P3 = Vd.transpose() * P1;
    Mat P4 = Ud_1.transpose() * P2;

    Mat Q1 = 0.5 * (P3 + P4);
    Mat Q2 = 0.5 * (P3 - P4);

    Mat Ud_2 = (m > n) ? Ud.rightCols(m - n) : Mat(m, 0);
    Mat E3, E4, E5;
    if (m > n) {
        Mat Q3 = (1.0 / sqrt2) * P.transpose() * Ud_2;
        Mat Q4 = (1.0 / sqrt2) * Ud_2.transpose() * P2;

        bool near_singular = false;
        for (int i = 0; i < n; ++i) {
            if (std::abs(Sigma_n(i, i)) < T(1e-14)) {
                near_singular = true;
                break;
            }
        }
        if (near_singular) {
            std::cerr << "Warning: Sigma_n nearly singular during E3/E4\n";
            return;
        }

        E3 = Sigma_n.inverse() * Q3;
        E4 = Q4 * Sigma_n.inverse();
        
        E5 = 0.5 * (Mat::Identity(m - n, m - n) - Ud_2.transpose() * Ud_2);
    }

    Mat E1 = Mat::Zero(n, n), E2 = Mat::Zero(n, n);
    Matrix<T, Dynamic, 1> sigma = Sigma_n.diagonal();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;  // важное условие
            T d1 = sigma(j) - sigma(i);
            T d2 = sigma(j) + sigma(i);

            // устойчивость: если разность мала — пропускаем (или задаем 0)
            if (std::abs(d1) < 1e-12) continue;
            if (std::abs(d2) < 1e-12) continue;

            E1(i, j) = Q1(i, j) / d1;
            E2(i, j) = Q2(i, j) / d2;
        }
    }
    E1.diagonal().setConstant(T(0.5));
    E2.diagonal().setZero();

    Mat U1 = Ud_1 + Ud_1 * (E1 - E2);
    if (m > n) U1 += sqrt2 * Ud_2 * E4;

    Mat U_new = Mat::Identity(m, m);
    U_new.leftCols(n) = U1;

    if (m > n) {
        Mat U2 = Ud_2 + Ud_2 * E5 - sqrt2 * Ud_1 * E3;

        // Реортогонализация
        HouseholderQR<Mat> qrU2(U2);
        U2 = qrU2.householderQ() * Mat::Identity(m, m - n);

        U_new.rightCols(m - n) = U2;
    }

    Mat V_new = Vd + Vd * (E1 + E2);

    HouseholderQR<Mat> qrU(U_new);
    U_new = qrU.householderQ();
    HouseholderQR<Mat> qrV(V_new);
    V_new = qrV.householderQ();

    if (!U_new.allFinite() || !V_new.allFinite()) {
        std::cerr << "Warning: U or V contains NaNs or Infs after update\n";
        return;
    }

    U = U_new;
    V = V_new;
}
