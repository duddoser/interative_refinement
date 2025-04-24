#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

template<typename K, int M, int N>
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
    Ogita_Aishima_SVD OA_SVD(const Matrix<T, M, N>& A, const  Matrix<T, M, M>& Ui, const  Matrix<T, N, N>& Vi) {

        const int m = A.rows();
        const int n = A.cols();

        if (m < n)
        {
            cout << "Attention! Number of the rows must be greater or equal  than  number of the columns";
            Ogita_Aishima_SVD ANS;
            return ANS;
        }

        //using matrix_dd = Matrix<double, Dynamic, Dynamic>;
        using matrix_nn = Matrix<double, N, N>; //V
        using matrix_mn = Matrix<double, M, N>; //A
        using matrix_mm = Matrix<double, M, M>; //U

        // Изменяем типы на double
        matrix_mn Ad = A.template cast<double>(); // Здесь матрицы A,U,V с элементами приведёнными к типу double
        matrix_mm Ud = Ui.template cast<double>();
        matrix_nn Vd = Vi.template cast<double>();

        // X - динамический размер. d - тип double
        MatrixXd Im = MatrixXd::Identity(m, m); // Единичная матрица m x m
        MatrixXd In = MatrixXd::Identity(n, n); // Единичная матрица n x n

        matrix_mm R = Im - Ud.transpose() * Ud;
        matrix_nn S = In -  Vd.transpose() * Vd;

        matrix_mn Tmn = Ud.transpose()*Ad*Vd;

        matrix_nn Sigma_n(n, n);
        Sigma_n.setZero();

        for (int i = 0; i < n; i++)
        {
            Sigma_n(i, i) = Tmn(i,i) / (1 - ((R(i,i) + S(i,i)) * 0.5));
        };

        matrix_mn Sigma;//Матрица сингулярных значений
        Sigma.setZero();
        Sigma.block(0, 0, n, n) = Sigma_n;

        matrix_nn R11 = R.topLeftCorner(n, n); // Верхний левый блок матрицы R размера n x n
        matrix_nn T1 = Tmn.topRows(n); // Квадратная матрица n x n из первых n строк матрицы T

        matrix_nn Calpha = T1 + R11 * Sigma_n;
        matrix_nn Cbeta = T1.transpose() + S * Sigma_n;
        matrix_nn D = Sigma_n*Calpha + Cbeta*Sigma_n;
        matrix_nn E = Calpha*Sigma_n + Sigma_n*Cbeta;

        matrix_nn G;
        matrix_mm F;

        double temp1;
        double temp2;
/*
        for (int i = 0; i < n; i++)
        {
            temp1 = Sigma_n(i,i) * Sigma_n(i,i);
            for (int j = 0; j < n; j++)
            {
                temp2 = Sigma_n(j,j) * Sigma_n(j,j);
                G(i,j) = D(i,j) / (temp2 - temp1);
                F(i,j) = E(i,j) / (temp2 - temp1);
            }
        }
*/
        for (int i = 0; i < n; i++)
        {
            temp1 = Sigma_n(i, i) * Sigma_n(i, i);
            for (int j = 0; j < n; j++)
            {
                if (i != j) {
                    temp2 = Sigma_n(j, j) * Sigma_n(j, j);
                    G(i, j) = D(i, j) / (temp2 - temp1);
                    F(i, j) = E(i, j) / (temp2 - temp1);
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = n; j < m; j++)
            {
                 F(i,j) = -Tmn(j,i)/Sigma_n(i,i);
            }
        }
        for (int i = n; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                F(i,j) = R(i,i)-F(j,i);
            }
        }
        for (int i = 0; i < n; i++)
        {
            G(i,i) = S(i,i) * 0.5;
            F(i,i) = R(i,i) * 0.5;
        }
        for (int i = n; i < m; i++)
        {
            F(i,i) = R(i,i) * 0.5;
        }
        matrix_mm U = Ud + Ud * F;//Вычисление уточнённых значений левых сингулярных векторов
        matrix_nn V = Vd + Vd * G;//Вычисление уточнённых значений правых сингулярных векторов

        Ogita_Aishima_SVD<T,M,N> ANS;
        ANS.Set_V(V.template cast<T>());// Приведение матриц к изначальному типу
        ANS.Set_U(U.template cast<T>());
        ANS.Set_S(Sigma.template cast<T>());
        return ANS;
    }
public:
    Ogita_Aishima_SVD(){};
    Ogita_Aishima_SVD(Matrix<T, M, N> A)
    {
        // Здесь должен использоваться неточный расчёт левых и правых сингулярных векторов с помощью функции из библиотеки Eigen
        BDCSVD<Matrix<T, Dynamic, Dynamic>> svd(A, ComputeFullU | ComputeFullV);
        Ogita_Aishima_SVD<T, M, N> temp = OA_SVD(A, svd.matrixU(), svd.matrixV());// Уточнение результата нашей функцией

        this->U = temp.matrixU();
        this->V = temp.matrixV();
        this->S = temp.singularValues();
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
