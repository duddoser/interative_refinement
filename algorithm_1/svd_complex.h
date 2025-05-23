#include <iostream>
#include <iomanip>
#include <cmath>
#include <tuple>
#include <vector>
#include <complex>

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
using namespace Eigen;

template<typename Scalar, int M, int N>
class SVD;

template<typename Scalar, int M, int N>
class SVD {
private:
    Eigen::Matrix<Scalar, M, M> U;
    Eigen::Matrix<Scalar, M, N> S;
    Eigen::Matrix<Scalar, N, N> V;

    void Set_U(const Eigen::Matrix<Scalar, M, M>& a) { U = a; };
    void Set_S(const Eigen::Matrix<Scalar, M, N>& a) { S = a; };
    void Set_V(const Eigen::Matrix<Scalar, N, N>& a) { V = a; };

protected:
    // Работает для любого Scalar, в том числе и для комплексных.
    // Для комплексного типа преобразования делаются через cast в double,
    // но если алгоритм требует модификаций для комплексных операций, нужно их добавить.
    SVD RefSVD(const Eigen::Matrix<Scalar, M, N>& A,
               const Eigen::Matrix<Scalar, M, M>& Ui,
               const Eigen::Matrix<Scalar, N, N>& Vi)
    {
        const int m = A.rows();
        const int n = A.cols();

        if (m < n)
        {
            cout << "Attention! Number of the rows must be greater or equal than number of the columns";
            SVD ans;
            return ans;
        }

        using matrix_dd = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using matrix_mm = Eigen::Matrix<Scalar, M, M>; // U
        using matrix_mn = Eigen::Matrix<Scalar, M, N>; // A
        using matrix_nn = Eigen::Matrix<Scalar, N, N>; // V
        
        // Приводим матрицы к типу Scalar (может быть комплексным или вещественным)
        matrix_mn Ad = A.template cast<Scalar>();
        matrix_mm Ud = Ui.template cast<Scalar>();
        matrix_nn Vd = Vi.template cast<Scalar>();

        // Step 1: Compute temp matrices: R, S, T.
        // Здесь R и S рассчитываются как (I - U^H * U) и (I - V^H * V).
        matrix_mm R = matrix_dd::Identity(m, m) - Ud.adjoint() * Ud;
        matrix_nn S_mat = matrix_dd::Identity(n, n) - Vd.adjoint() * Vd;
        matrix_mn T = Ud.adjoint() * Ad * Vd;

        matrix_nn F11 = matrix_nn::Zero(n, n);
        matrix_nn G = matrix_nn::Zero(n, n);

        // Step 2 and 3: compute diag parts.
        matrix_nn Sigma_n = matrix_nn::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            // Здесь деление производится по формуле, предполагающей вещественную арифметику.
            // При работе с комплексными может потребоваться поправка.
            Sigma_n(i, i) = T(i, i) / ( static_cast<Scalar>(1.0) - (R(i, i) + S_mat(i, i)) * static_cast<Scalar>(0.5) );
            F11(i, i) = R(i, i) * static_cast<Scalar>(0.5);
            G(i, i) = S_mat(i, i) * static_cast<Scalar>(0.5);
        }
        
        // Step 4: Compute off-diagonal parts of F11 and G.
        Scalar alpha, betta;
        Scalar sigma_i_sqr, sigma_j_sqr;
        for (int i = 0; i < n; ++i) {
            sigma_i_sqr = norm(Sigma_n(i, i)); // для комплексного числа используем norm (квадрат модуля)
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    sigma_j_sqr = norm(Sigma_n(j, j));
                    alpha = T(i, j) + Sigma_n(j, j) * R(i, j);
                    betta = T(j, i) + Sigma_n(j, j) * S_mat(i, j);
                    F11(i, j) = (alpha * Sigma_n(j, j) + betta * Sigma_n(i, i))
                                  / (static_cast<Scalar>(sigma_j_sqr) - static_cast<Scalar>(sigma_i_sqr));
                    G(i, j) = (alpha * Sigma_n(i, i) + betta * Sigma_n(j, j))
                               / (static_cast<Scalar>(sigma_j_sqr) - static_cast<Scalar>(sigma_i_sqr));

                }
            }
        }

        // Step 5: сбор диагональной матрицы Sigma (размер m x n)
        matrix_mn Sigma = matrix_dd::Zero(m, n);
        Sigma.block(0, 0, n, n) = Sigma_n.adjoint(); 

        // Step 6: compute F12;
        Eigen::Matrix<Scalar, Dynamic, Dynamic> F12(n, m - n);
        for (int i = 0; i < n; ++i) {
            for (int j = n; j < m; ++j) {
                F12(i, j - n) = -T(j, i) / Sigma_n(i, i);
            }
        }
        
        // Step 7: compute F21;
        Eigen::Matrix<Scalar, Dynamic, Dynamic> F21(m - n, n);
        for (int i = n; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                F21(i - n, j) = R(i, j) - F12(j, i - n);
            }
        }
        
        // Step 8: compute F22;
        Eigen::Matrix<Scalar, Dynamic, Dynamic> F22(m - n, m - n);
        for (int i = n; i < m; ++i) {
            for (int j = n; j < m; ++j) {
                F22(i - n, j - n) = R(i, j) * 0.5;
            }
        }
        
        // Compose F:
        matrix_mm F = matrix_dd::Zero(m, m);
        F.block(0, 0, n, n) = F11;
        F.block(0, n, n, m - n) = F12;
        F.block(n, 0, m - n, n) = F21;
        F.block(n, n, m - n, m - n) = F22;

        matrix_mm U_new = Ud + Ud * F;
        matrix_nn V_new = Vd + Vd * G;

        SVD<Scalar, M, N> ans;
        ans.Set_U(U_new.template cast<Scalar>());
        ans.Set_V(V_new.template cast<Scalar>());
        ans.Set_S(Sigma.template cast<Scalar>());
        return ans;
    };

public:
    SVD() {};

    SVD(Eigen::Matrix<Scalar, M, N> A)
    {
        // Здесь для разложения используем BDCSVD для вещественных данных - он не работает с комплексными.
        // Для комплексного варианта лучше использовать JacobiSVD. Поэтому, если Scalar комплексное,
        // применим SVD с помощью JacobiSVD.
        SVD<Scalar, M, N> temp;
        if constexpr (is_same<Scalar, std::complex<float>>::value || is_same<Scalar, std::complex<double>>::value) {
            Eigen::JacobiSVD< Eigen::Matrix<Scalar, Dynamic, Dynamic> > svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            temp = RefSVD(A, svd.matrixU(), svd.matrixV());
        }
        else {
            Eigen::BDCSVD< Eigen::Matrix<Scalar, Dynamic, Dynamic> > svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            temp = RefSVD(A, svd.matrixU(), svd.matrixV());
        }
        this->U = temp.matrixU();
        this->V = temp.matrixV();
        this->S = temp.singularValues();
    }

    Eigen::Matrix<Scalar, N, N> matrixV()
    {
        return V;
    }

    Eigen::Matrix<Scalar, M, M> matrixU()
    {
        return U;
    }

    Eigen::Matrix<Scalar, M, N> singularValues()
    {
        return S;
    }
};
