#include <iostream>
#include <iomanip>
#include <cmath>
#include <tuple>
#include <vector>
#if __has_include(<Eigen/LU>)
#include <Eigen/LU> 
#elif __has_include(<eigen3/Eigen/LU>)
#include <eigen3/Eigen/LU>
#endif

#if __has_include(<Eigen/SVD>)
#include <Eigen/SVD> 
#elif __has_include(<eigen3/Eigen/SVD>)
#include <eigen3/Eigen/SVD>
#endif

using namespace std;

template<typename Scalar, int M, int N>
class SVD;

template<typename Scalar, int M, int N>
class SVD {
private:
    Eigen::Matrix<Scalar, M, M> U; 
    Eigen::Matrix<Scalar, M, N> S; 
    Eigen::Matrix<Scalar, N, N> V; 

    void Set_U(const Eigen::Matrix<Scalar, M, M>& a) { U = a; }
    void Set_S(const Eigen::Matrix<Scalar, M, N>& a) { S = a; }
    void Set_V(const Eigen::Matrix<Scalar, N, N>& a) { V = a; }

protected:
    SVD RefSVD(const Eigen::Matrix<Scalar, M, N>& A,
               const Eigen::Matrix<Scalar, M, M>& Ui,
               const Eigen::Matrix<Scalar, N, N>& Vi) {
        const int m = A.rows();
        const int n = A.cols(); 

        if (m < n) {
            cout << "Attention! Number of the rows must be greater or equal than number of the columns" << endl;
            SVD ans;
            return ans;
        }

        using matrix_dd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        using matrix_mm = Eigen::Matrix<double, M, M>;
        using matrix_mn = Eigen::Matrix<double, M, N>;
        using matrix_nn = Eigen::Matrix<double, N, N>;

        matrix_mn Ad = A.template cast<double>();
        matrix_mm Ud = Ui.template cast<double>();
        matrix_nn Vd = Vi.template cast<double>();

        // Шаг 1. Вычисляем промежуточные матрицы:
        // R = I_m - Udᵀ * Ud, S = I_n - Vdᵀ * Vd, T = Udᵀ * Ad * Vd.
        matrix_mm R = matrix_dd::Identity(m, m) - Ud.transpose() * Ud;
        matrix_nn S_mat = matrix_dd::Identity(n, n) - Vd.transpose() * Vd;
        matrix_mn T = Ud.transpose() * Ad * Vd;

        // Шаг 2. Инициализируем F11 (N×N) и G (N×N)
        matrix_nn F11 = matrix_nn::Zero(n, n);
        matrix_nn G = matrix_nn::Zero(n, n);

        // Инициализируем матрицу Sigma_n (N×N) для уточнённых сингулярных значений
        matrix_nn Sigma_n = matrix_nn::Zero(n, n);

        // Шаг 3. Вычисляем диагональные элементы:
        for (int i = 0; i < n; i++) {
            Sigma_n(i, i) = T(i, i) / (1.0 - 0.5 * (R(i, i) + S_mat(i, i)));
            F11(i, i) = 0.5 * R(i, i);
            G(i, i) = 0.5 * S_mat(i, i);
        }

        // Шаг 4. Вычисляем внедиагональные элементы F11 и G:
        double alpha, betta, sigma_i_sqr, sigma_j_sqr;
        for (int i = 0; i < n; i++) {
            sigma_i_sqr = Sigma_n(i, i) * Sigma_n(i, i);
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sigma_j_sqr = Sigma_n(j, j) * Sigma_n(j, j);
                    alpha = T(i, j) + Sigma_n(j, j) * R(i, j);
                    betta = T(j, i) + Sigma_n(j, j) * S_mat(i, j);
                    F11(i, j) = (alpha * Sigma_n(j, j) + betta * Sigma_n(i, i)) / (sigma_j_sqr - sigma_i_sqr);
                    G(i, j) = (alpha * Sigma_n(i, i) + betta * Sigma_n(j, j)) / (sigma_j_sqr - sigma_i_sqr);
                }
            }
        }

        // Шаг 5. Формируем матрицу сингулярных значений Sigma (размер M×N).
        // Записываем диагональный блок Sigma_n (N×N) в верхний левый угол матрицы Sigma.
        matrix_mn Sigma = matrix_mn::Zero(m, n);
        Sigma.block(0, 0, n, n) = Sigma_n.transpose();

        // Шаг 6. Вычисляем минор F12 (размер N×(m - n)):
        matrix_dd F12(n, m - n);
        for (int i = 0; i < n; i++) {
            for (int j = n; j < m; j++) {
                F12(i, j - n) = -T(j, i) / Sigma_n(i, i);
            }
        }

        // Шаг 7. Вычисляем минор F21 (размер (m - n)×N):
        matrix_dd F21(m - n, n);
        for (int i = n; i < m; i++) {
            for (int j = 0; j < n; j++) {
                F21(i - n, j) = R(i, j) - F12(j, i - n);
            }
        }

        // Шаг 8. Вычисляем минор F22 (размер (m - n)×(m - n)):
        matrix_dd F22(m - n, m - n);
        for (int i = n; i < m; i++) {
            for (int j = n; j < m; j++) {
                F22(i - n, j - n) = 0.5 * R(i, j);
            }
        }

        // Шаг 9. Собираем полную матрицу коррекции F (размер M×M)
        matrix_mm F = matrix_mm::Zero();
        F.block(0, 0, n, n) = F11;
        F.block(0, n, n, m - n) = F12;
        F.block(n, 0, m - n, n) = F21;
        F.block(n, n, m - n, m - n) = F22;

        // Шаг 10. Вычисляем уточнённые матрицы U и V:
        matrix_mm U_new = Ud + Ud * F;
        matrix_nn V_new = Vd + Vd * G;

        // Собираем результат: приводим обратно к исходному типу Scalar
        SVD<Scalar, M, N> ans;
        ans.Set_U(U_new.template cast<Scalar>());
        ans.Set_V(V_new.template cast<Scalar>());
        ans.Set_S(Sigma.template cast<Scalar>());
        return ans;
    }

public:
    SVD() {}

    SVD refine(const Eigen::Matrix<Scalar, M, N>& A) const {
        return RefSVD(A, this->matrixU(), this->matrixV());
    }

    // Конструктор, принимающий матрицу A
    SVD(Eigen::Matrix<Scalar, M, N> A) {
        Eigen::BDCSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
            svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        SVD<Scalar, M, N> temp = RefSVD(A, svd.matrixU(), svd.matrixV());
        this->U = temp.matrixU();
        this->V = temp.matrixV();
        this->S = temp.singularValues();
    }

    Eigen::Matrix<Scalar, N, N> matrixV() { return V; }
    Eigen::Matrix<Scalar, M, M> matrixU() { return U; }
    Eigen::Matrix<Scalar, M, N> singularValues() { return S; }

    static void refine(const Eigen::Matrix<Scalar, M, N>& A,
            Eigen::Matrix<Scalar, M, M>& U,
            Eigen::Matrix<Scalar, N, N>& V,
            Eigen::Matrix<Scalar, M, N>& S) {
        SVD<Scalar, M, N> wrapper;
        auto result = wrapper.RefSVD(A, U, V);
        U = result.matrixU();
        V = result.matrixV();
        S = result.singularValues();
    }
};
