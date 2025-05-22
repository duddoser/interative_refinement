// Название статьи:  Uchino Yuki, Terao Takeshi, Ozaki Katsuhisa. 2022.08.05 – Acceleration of
// Iterative Refinement for Singular Value Decomposition.
// (https://www.researchgate.net/publication/362642883_Acceleration_of_Iterative_Refinement_for_Singular_Value_Decomposition)
// Обязательные условия: 
// 1) В матрицах количествово строк больше или равному количеству столбцов
// 2) каждое сингулярное значение меньше предыдущего (d1>d2>...>dn)
//  каждое приближенное сингулярное значение отлично друг от друга (di != dj для i!=j)
//
// Выполнил Минаков В.С. КМБО-01-23
// 

#ifndef OGITA_AISHIMA_SVD_H
#define OGITA_AISHIMA_SVD_H

#include <iostream>
#include "eigen/Eigen/Dense"

using namespace std;
using namespace Eigen;

template<typename T, int M, int N>
class Ogita_Aishima_SVD;

template<typename T, int M, int N>
class Ogita_Aishima_SVD {
private:
    Matrix<T, M, M> U;
    Matrix<T, M, N> S;
    Matrix<T, N, N> V;

    void Set_U(Matrix<T, M, M> A)
    {
        U = A;
    }
    void Set_V(Matrix<T, N, N> A)
    {
        V = A;
    }
    void Set_S(Matrix<T, M, N> A)
    {
        S = A;
    }
protected:
    Ogita_Aishima_SVD OA_SVD(const Matrix<T, M, N>& A, const Matrix<T, M, M>& Ui, const Matrix<T, N, N>& Vi) const 
    {
        const int m = A.rows();
        const int n = A.cols();

        if (m < n) 
        {
            cout << "Attention! Number of the rows must be greater or equal than number of the columns" << endl;
            return Ogita_Aishima_SVD();
        }

        using matrix = Matrix<long double, Dynamic, Dynamic>;

        // Приведение к long double (Matrix<long double,Dynamic,Dynamic>)
        matrix Ad = A.template cast<long double>();
        matrix Ud = Ui.template cast<long double>();
        matrix Vd = Vi.template cast<long double>();

        // Единичные матрицы
        matrix Im = Matrix<long double, Dynamic, Dynamic>::Identity(m, m);
        matrix In = Matrix<long double, Dynamic, Dynamic>::Identity(n, n);
        
        matrix R = Im - Ud.transpose() * Ud;
        matrix S_ = In - Vd.transpose() * Vd;
        matrix Tmn = Ud.transpose() * Ad * Vd;

        // Матрица сингулярных значений
        matrix Sigma_n = matrix::Zero(n, n);
        
        for (int i = 0; i < n; ++i) 
            Sigma_n(i,i) = Tmn(i,i) / (1.0 - (( R(i,i) + S_(i,i) ) * 0.5 ));  

        // Расширяем до m×n
        matrix Sigma = matrix::Zero(m, n);
        Sigma.block(0, 0, n, n) = Sigma_n;

        matrix R11 = R.topLeftCorner(n, n); // Верхний левый блок матрицы R размера n x n
        matrix T1  = Tmn.topRows(n); // Квадратная матрица n x n из первых n строк матрицы T
        matrix Calpha = T1 + R11 * Sigma_n;
        matrix Cbeta = T1.transpose() + S_ * Sigma_n;

        matrix D = Sigma_n * Calpha + Cbeta * Sigma_n;
        matrix E = Calpha * Sigma_n + Sigma_n * Cbeta;

        matrix G = matrix::Zero(n, n);
        matrix F = matrix::Zero(m, m);

        long double temp1, temp2;

        for (int i = 0; i < n; ++i)
        {
            temp1 = Sigma_n(i, i) * Sigma_n(i, i);
            for (int j = 0; j < n; ++j)
            {
                if (i != j) {
                    temp2 = Sigma_n(j, j) * Sigma_n(j, j);
                    G(i, j) = D(i, j) / (temp2 - temp1);
                    F(i, j) = E(i, j) / (temp2 - temp1);
                }
            }
        }

        for (int i = 0; i < n; ++i)
            for (int j = n; j < m; ++j)
                 F(i,j) = -Tmn(j,i)/Sigma_n(i,i);
        
        for (int i = n; i < m; ++i)
            for (int j = 0; j < n; ++j)
                F(i,j) = R(i,j)-F(j,i);

        for (int i = 0; i < n; ++i)
        {
            G(i,i) = S_(i,i) * 0.5;
            F(i,i) = R(i,i) * 0.5;
        }

        for (int i = n; i < m; ++i)
            for (int j = n; j < m; ++j)
                F(i,j) = R(i,j) * 0.5;


        matrix U = Ud + Ud * F;//Вычисление уточнённых значений левых сингулярных векторов
        matrix V = Vd + Vd * G;//Вычисление уточнённых значений правых сингулярных векторов

        Ogita_Aishima_SVD<T, M, N> ANS;
        ANS.Set_V(V.template cast<T>());// Приведение матриц к изначальному типу
        ANS.Set_U(U.template cast<T>());
        ANS.Set_S(Sigma.template cast<T>());
        return ANS;
    }
    

public:
    Ogita_Aishima_SVD() {};
    // стандартный SVD + одна итерация уточнения
    Ogita_Aishima_SVD(Matrix<T, M, N> A) {
        BDCSVD<Matrix<T,Dynamic,Dynamic>> svd(A, ComputeFullU | ComputeFullV);
        Ogita_Aishima_SVD<T,M,N> temp = OA_SVD(A, svd.matrixU(), svd.matrixV());
        U = temp.matrixU();
        V = temp.matrixV();
        S = temp.singularValues();
    }
    // итеративное уточнение от заданных U,V
    Ogita_Aishima_SVD(Matrix<T, M, N> A, Matrix<T, M, M> Ui, Matrix<T, N, N> Vi, int iterations) {
        Ogita_Aishima_SVD<T,M,N> current = OA_SVD(A, Ui, Vi);
        for (int k = 1; k < iterations; ++k)
            current = OA_SVD(A, current.matrixU(), current.matrixV());
        U = current.matrixU();
        V = current.matrixV();
        S = current.singularValues();
    }

    Matrix<T, N, N> matrixV()
    {
        return V;
    }

    Matrix<T, M, M> matrixU()
    {
        return U;
    }

    Matrix<T, M, N> singularValues()
    {
        return S;
    }
};

#endif // OGITA_AISHIMA_SVD_H
