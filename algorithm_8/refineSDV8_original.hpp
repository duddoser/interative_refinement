#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

template<typename T, int M, int N>
class Ogita_Aishima_SVD {
private:
    Matrix<T, M, M> U;
    Matrix<T, M, N> S;
    Matrix<T, N, N> V;

    void Set_U(Matrix<T, M, M> A) { U = A; }
    void Set_V(Matrix<T, N, N> A) { V = A; }
    void Set_S(Matrix<T, M, N> A) { S = A; }

protected:
    Ogita_Aishima_SVD OA_SVD(const Matrix<T, M, N>& A, const Matrix<T, M, M>& Ui, const Matrix<T, N, N>& Vi) {
        const int m = A.rows();
        const int n = A.cols();

        if (m < n) {
            std::cout << "Error: Number of rows must be >= number of columns";
            return Ogita_Aishima_SVD();
        }

        using matrix_nn = Matrix<double, N, N>;
        using matrix_mn = Matrix<double, M, N>;
        using matrix_mm = Matrix<double, M, M>;
        using matrix_nm = Matrix<double, N, M>;

        matrix_mn Ad = A.template cast<double>();
        matrix_mm Ud = Ui.template cast<double>();
        matrix_nn Vd = Vi.template cast<double>();

        MatrixXd Im = MatrixXd::Identity(m, m);
        MatrixXd In = MatrixXd::Identity(n, n);

        matrix_mn Ud_1 = Ud.block(0, 0, m, n);

        matrix_mn P = Ad * Vd;
        matrix_nn Q = Ad.transpose() * Ud_1;

        matrix_nn ViT = Vd.transpose();
        matrix_mm UiT = Ud.transpose();

        std::vector<double> r(n, 0.0);
        std::vector<double> t(n, 0.0);
        matrix_nn Sigma_n(n, n);
        Sigma_n.setZero();

        for (int i = 0; i < n; i++) {
            double ui_ud = (UiT.row(i) * Ud.col(i))(0, 0);
            double vi_vd = (ViT.row(i) * Vd.col(i))(0, 0);
            r[i] = 1.0 - 0.5 * (ui_ud + vi_vd);
            t[i] = UiT.row(i) * P.col(i);
            Sigma_n(i, i) = t[i] / (1 - r[i]);
        }

        matrix_mn Sigma(m, n);
        Sigma.setZero();
        Sigma.block(0, 0, n, n) = Sigma_n;

        matrix_nn P1 = Q - Vd * Sigma_n;
        matrix_mn P2 = P - Ud_1 * Sigma_n;

        matrix_nn P3 = Vd.transpose() * P1;
        matrix_nn P4 = Ud_1.transpose() * P2;

        matrix_nn Q1 = 0.5 * (P3 + P4);
        matrix_nn Q2 = 0.5 * (P3 - P4);

        // Объявляем переменные до условий
        MatrixXd Ud_2;
        MatrixXd E3, E4, E5;
        
        if (m > n) {
            Ud_2 = Ud.block(0, n, m, m - n);
            
            MatrixXd Q3 = (1.0 / std::sqrt(2.0)) * P.transpose() * Ud_2;
            MatrixXd Q4 = (1.0 / std::sqrt(2.0)) * Ud_2.transpose() * P2;
            
            E3 = Sigma_n.inverse() * Q3;
            E4 = Q4 * Sigma_n.inverse();
            E5 = 0.5 * (MatrixXd::Identity(m-n, m-n) - Ud_2.transpose() * Ud_2);
        } else {
            // Инициализация нулевыми матрицами соответствующего размера
            Ud_2 = MatrixXd::Zero(m, 0);
            E3 = MatrixXd::Zero(n, 0);
            E4 = MatrixXd::Zero(0, n);
            E5 = MatrixXd::Zero(0, 0);
        }

        matrix_nn E1 = matrix_nn::Zero();
        matrix_nn E2 = matrix_nn::Zero();

        Eigen::VectorXd sigma = Sigma_n.diagonal();
        Eigen::MatrixXd denom1 = sigma.replicate(1, n) - sigma.transpose().replicate(n, 1);
        Eigen::MatrixXd denom2 = sigma.replicate(1, n) + sigma.transpose().replicate(n, 1);

        E1 = Q1.cwiseQuotient(denom1);
        E2 = Q2.cwiseQuotient(denom2);

        for (int i = 0; i < n; ++i)
            E1(i, i) = 0.5 * r[i];
        E2.diagonal().setZero();

        // Исправленная часть: формирование матрицы U
        matrix_mn U1 = Ud_1 + Ud_1 * (E1 - E2);
        if (m > n) {
            U1 += std::sqrt(2.0) * Ud_2 * E4;
        }

        matrix_mm U = matrix_mm::Identity(m, m);
        U.block(0, 0, m, n) = U1;

        if (m > n) {
            MatrixXd U2 = Ud_2 + Ud_2 * E5 - std::sqrt(2.0) * Ud_1 * E3;
            U.block(0, n, m, m - n) = U2;
        }

        matrix_nn V_new = Vd + Vd * (E1 + E2);

        Ogita_Aishima_SVD<T, M, N> ANS;
        ANS.Set_U(U.template cast<T>());
        ANS.Set_V(V_new.template cast<T>());
        ANS.Set_S(Sigma.template cast<T>());

        return ANS;
    }

public:
    Ogita_Aishima_SVD() {}

    Ogita_Aishima_SVD(Matrix<T, M, N> A) {
        BDCSVD<Matrix<T, Dynamic, Dynamic>> svd(A, ComputeFullU | ComputeFullV);
        
        Ogita_Aishima_SVD<T, M, N> temp = OA_SVD(A, svd.matrixU(), svd.matrixV());

        this->U = temp.matrixU();
        this->V = temp.matrixV();
        this->S = temp.singularValues();
    }

    Matrix<T, M, M> matrixU() { return U; }
    Matrix<T, N, N> matrixV() { return V; }
    Matrix<T, M, N> singularValues() { return S; }


    static void refineSVD8(const Matrix<T, M, N>& A,
        Matrix<T, M, M>& U,
        Matrix<T, N, N>& V,
        Matrix<T, M, N>& S) {

            Ogita_Aishima_SVD<T, M, N> wrapper;
            auto result = wrapper.OA_SVD(A, U, V);
            U = result.matrixU();
            V = result.matrixV();
            S = result.singularValues();
    }
};