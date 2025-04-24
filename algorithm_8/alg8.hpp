#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// Шаблонный класс для итеративного уточнения SVD по алгоритму Ogita-Aishima (Algorithm 8)
template<typename T, int M, int N>
class Ogita_Aishima_SVD {
private:
    Matrix<T, M, M> U;  // Левые сингулярные векторы (m x m)
    Matrix<T, M, N> S;  // Матрица сингулярных значений (m x n)
    Matrix<T, N, N> V;  // Правые сингулярные векторы (n x n)

    // Сеттеры для матриц U, V, S
    void Set_U(Matrix<T, M, M> A) { U = A; }
    void Set_V(Matrix<T, N, N> A) { V = A; }
    void Set_S(Matrix<T, M, N> A) { S = A; }

protected:
    // Основная функция итеративного уточнения SVD
    Ogita_Aishima_SVD OA_SVD(const Matrix<T, M, N>& A, const Matrix<T, M, M>& Ui, const Matrix<T, N, N>& Vi) {
        // Проверка размерности: число строк должно быть >= числа столбцов
        if (A.rows() < A.cols()) {
            std::cout << "Attention! Number of the rows must be greater or equal than number of the columns";
            return Ogita_Aishima_SVD();  // Возвращаем пустой объект в случае ошибки
        }

        const int m = A.rows();
        const int n = A.cols();

        // Определение типов матриц для удобства
        using matrix_nn = Matrix<double, N, N>;
        using matrix_mn = Matrix<double, M, N>;
        using matrix_mm = Matrix<double, M, M>;
        using matrix_nm = Matrix<double, N, M>;
        using matrix_m_mn = Matrix<double, M, M-N>;
        using matrix_n_mn = Matrix<double, N, M-N>;
        using matrix_mn_m = Matrix<double, M-N, M>;
        using matrix_mn_n = Matrix<double, M-N, N>;
        using matrix_mn_mn = Matrix<double, M-N, M-N>;

        // Приведение входных матриц к double для точных вычислений
        matrix_mn Ad = A.template cast<double>();
        matrix_mm Ud = Ui.template cast<double>();
        matrix_nn Vd = Vi.template cast<double>();
        
        // Единичные матрицы
        MatrixXd Im = MatrixXd::Identity(m, m);
        MatrixXd In = MatrixXd::Identity(n, n);

        // Блок левых сингулярных векторов (первые n столбцов)
        matrix_mn Ud_1 = Ud.block(0, 0, m, n);

        // Шаг 1: Вычисление промежуточных матриц P и Q
        matrix_mn P = Ad * Vd;  // P = A * V
        matrix_nn Q = Ad.transpose() * Ud_1;  // Q = A^T * U_1

        // Транспонированные версии V и U
        matrix_nn ViT = Vd.transpose();
        matrix_mm UiT = Ud.transpose();

        // Векторы для хранения промежуточных значений
        std::vector<double> r(n, 0.0);  // Невязки ортогональности
        std::vector<double> t(n, 0.0);  // Временные значения для сингулярных чисел
        matrix_nn Sigma_n(n, n);  // Матрица сингулярных значений (n x n)
        Sigma_n.setZero();

        // Шаг 2: Вычисление приближенных сингулярных значений
        for (int i = 0; i < n; i++) {
            // Вычисление невязок ортогональности
            double ui_ud = (UiT.row(i) * Ud.col(i))(0, 0);
            double vi_vd = (ViT.row(i) * Vd.col(i))(0, 0);
            r[i] = 1.0 - 0.5 * (ui_ud + vi_vd);
            
            // Вычисление временных значений для сингулярных чисел
            t[i] = UiT.row(i) * P.col(i);
            Sigma_n(i, i) = t[i] / (1 - r[i]);  // Формула для сингулярных значений
        }

        // Расширение матрицы сингулярных значений до размеров m x n
        matrix_mn Sigma(m, n);
        Sigma.setZero();
        Sigma.block(0, 0, n, n) = Sigma_n;

        // Шаг 3: Вычисление поправочных матриц
        matrix_nn P1 = Q - Vd * Sigma_n;  // P1 = Q - V * Sigma_n
        matrix_mn P2 = P - Ud_1 * Sigma_n;  // P2 = P - U_1 * Sigma_n

        matrix_nn P3 = Vd.transpose() * P1;  // P3 = V^T * P1
        matrix_nn P4 = Ud_1.transpose() * P2;  // P4 = U_1^T * P2

        matrix_nn Q1 = 0.5 * (P3 + P4);  // Симметричная часть
        matrix_nn Q2 = 0.5 * (P3 - P4);  // Антисимметричная часть

        // Блок оставшихся сингулярных векторов (столбцы n..m-1)
        matrix_m_mn Ud_2 = Ud.block(0, n, m, m - n);

        // Шаг 4: Вычисление поправочных матриц для блоков U и V
        matrix_n_mn Q3 = 1.0 / std::sqrt(2.0) * P.transpose() * Ud_2;
        matrix_mn_n Q4 = 1.0 / std::sqrt(2.0) * Ud_2.transpose() * P2;

        // Инициализация поправочных матриц E1, E2
        matrix_nn E1 = matrix_nn::Zero();
        matrix_nn E2 = matrix_nn::Zero();

        // Вектор сингулярных значений
        Eigen::VectorXd sigma = Sigma_n.diagonal();

        // Вычисление знаменателей для E1 и E2
        Eigen::MatrixXd denom1 = sigma.replicate(1, n) - sigma.transpose().replicate(n, 1);
        Eigen::MatrixXd denom2 = sigma.replicate(1, n) + sigma.transpose().replicate(n, 1);

        // Заполнение E1 и E2
        E1 = Q1.cwiseQuotient(denom1);  // E1 = Q1 ./ (sigma_i - sigma_j)
        E2 = Q2.cwiseQuotient(denom2);  // E2 = Q2 ./ (sigma_i + sigma_j)

        // Установка диагонали E1 (поправка для ортогональности)
        for (int i = 0; i < n; ++i)
            E1(i, i) = 0.5 * r[i];

        E2.diagonal().setZero();  // Диагональ E2 обнуляется

        // Дополнительные поправочные матрицы
        matrix_n_mn E3 = Sigma_n.inverse() * Q3;
        matrix_mn_n E4 = Q4 * Sigma_n.inverse();
        matrix_mn_mn E5 = 0.5 * (MatrixXd::Identity(m-n, m-n) - Ud_2.transpose() * Ud_2);

        // Шаг 5: Обновление сингулярных векторов
        matrix_mn U1 = Ud_1 + Ud_1 * (E1 - E2) + std::sqrt(2.0) * Ud_2 * E4;

        //               m_mn   m_mn * mn_mn = m_mn   mn * n_mn = m_mn
        matrix_m_mn U2 = Ud_2 + Ud_2 * E5 - std::sqrt(2.0) * Ud_1 * E3;

        //               nn    nn * (nn + nn)
        matrix_nn V_new = Vd + Vd * (E1 + E2);

        // Объединение блоков U1 и U2 в полную матрицу U
        matrix_mm U;
        U << U1, U2;

        // Создание и возврат результата
        Ogita_Aishima_SVD<T, M, N> ANS;
        ANS.Set_U(U.template cast<T>());
        ANS.Set_V(V_new.template cast<T>());
        ANS.Set_S(Sigma.template cast<T>());

        return ANS;
    }

public:
    // Конструкторы
    Ogita_Aishima_SVD() {}  // Пустой конструктор

    // Основной конструктор: вычисляет SVD и сразу применяет итеративное уточнение
    Ogita_Aishima_SVD(Matrix<T, M, N> A) {
        // Вычисление начального приближения SVD с помощью Eigen
        BDCSVD<Matrix<T, Dynamic, Dynamic>> svd(A, ComputeFullU | ComputeFullV);
        
        // Применение итеративного уточнения
        Ogita_Aishima_SVD<T, M, N> temp = OA_SVD(A, svd.matrixU(), svd.matrixV());

        // Сохранение результатов
        this->U = temp.matrixU();
        this->V = temp.matrixV();
        this->S = temp.singularValues();
    }

    // Геттеры для результатов
    Matrix<T, M, M> matrixU() { return U; }
    Matrix<T, N, N> matrixV() { return V; }
    Matrix<T, M, N> singularValues() { return S; }
};
