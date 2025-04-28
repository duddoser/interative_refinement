#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Sparse"
#include <chrono>
using namespace std;
using namespace Eigen;
using namespace std::chrono;

vector<MatrixXd> Split_Mat(const MatrixXd& A, int l, double delta)
{
    const double u = std::numeric_limits<double>::epsilon();
    int m = A.rows();
    int n = A.cols();

    int k = 1;
    int beta = static_cast<int>(floor((-log2(u) + log2(n)) * 0.5));

    MatrixXd Acopy = A;

    vector<MatrixXd> D;
    D.push_back(MatrixXd::Zero(m, n));

    while (k < l)
    {
        VectorXd mu = Acopy.cwiseAbs().rowwise().maxCoeff();

        if (mu.maxCoeff() == 0.0)
        {
            return D;
        }

        VectorXd w = (mu.array().log2().ceil() + beta).exp2();

        MatrixXd S = w.replicate(1, n);

        MatrixXd Dk = (Acopy + S) - S;
        Acopy = Acopy - Dk;

        SparseMatrix<double> Dk_sparse = Dk.sparseView(); 
        if (Dk_sparse.nonZeros() < delta * m * n)  
        {
            Dk = MatrixXd(Dk_sparse);  
        }

        D.push_back(Dk);
        k++;
    }

    if (k == l)
        D.push_back(Acopy);

    return D;
}

MatrixXd Acc_Mul(const MatrixXd& A, const MatrixXd& B, int k = 2, double delta = 0.0001) 
{
    int m = A.rows(), n = A.cols(), p = B.cols();
    MatrixXd BT = B.transpose();

    vector<MatrixXd> D = Split_Mat(A, k, delta);
    vector<MatrixXd> E = Split_Mat(BT, k, delta);
    int hA = D.size(), hB = E.size();

    for (int i = 0; i < hB; i++) 
    {
        MatrixXd tmp = E[i].transpose();
        E[i] = tmp;
    }


    vector<MatrixXd> G;
    int l = 0;

    for (int r = 0; r < min(hA, k - 1); r++) {
        for (int s = 0; s < min(hB, k - 1); s++) 
        {
            if ((r + s) <= k) 
            {
                l++;
                G.push_back(D[r] * E[s]);
            }
        }
    }

    for (int r = 0; r < hA; r++) 
    {
        MatrixXd F = MatrixXd::Zero(n, p);
        for (int s = k - r; s < hB; s++)
            F = F + E[s];
        l++;
        G.push_back(D[r] * F);
    }

    MatrixXd C = G[0];
    for (int i = 1; i < l; ++i) {
        C = C + G[i];
    }

    return C;
}

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



        matrix_mm R = Im - Acc_Mul(Ud.transpose(), Ud);
        matrix_nn S = In - Acc_Mul(Vd.transpose(), Vd);
        matrix_mn Tmn = Acc_Mul(Ud.transpose(), Acc_Mul(Ad, Vd));

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

        matrix_nn Calpha = T1 + Acc_Mul(R11, Sigma_n);
        matrix_nn Cbeta = T1.transpose() + Acc_Mul(S, Sigma_n);
        matrix_nn D = Acc_Mul(Sigma_n, Calpha) + Acc_Mul(Cbeta, Sigma_n);
        matrix_nn E = Acc_Mul(Calpha, Sigma_n) + Acc_Mul(Sigma_n, Cbeta);

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

        matrix_mm U = Ud + Acc_Mul(Ud, F);//Вычисление уточнённых значений левых сингулярных векторов
        matrix_nn V = Vd + Acc_Mul(Vd, G);//Вычисление уточнённых значений правых сингулярных векторов

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
        JacobiSVD<Matrix<T, Dynamic, Dynamic>> svd(A, ComputeFullU | ComputeFullV);
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

template<int Rows, int Cols>
void run_test() {
    //string filename = "Jacobi_svd_test_tttttttttttttt" + to_string(Rows) + "x" + to_string(Cols) + ".txt";
    //ofstream fout(filename);
    //if (!fout.is_open()) {
    //    cerr << "Failed to open " << filename << endl;
    //    return;
    //}

    // Генерация случайной матрицы
    Matrix<float, Rows, Cols> A = Matrix<float, Rows, Cols>::Random();

    // SVD через Eigen
    JacobiSVD<Matrix<float, Rows, Cols>> svd(A, ComputeFullU | ComputeFullV);

    Matrix<float, Rows, Cols> Sigma = Matrix<float, Rows, Cols>::Zero();
    for (int i = 0; i < svd.singularValues().size(); ++i)
        Sigma(i, i) = svd.singularValues()(i);

    Matrix<float, Rows, Cols> A_bdc = svd.matrixU() * Sigma * svd.matrixV().transpose();

    // Наш алгоритм
    auto start = high_resolution_clock::now();
    Ogita_Aishima_SVD<float, Rows, Cols> ans(A);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    Matrix<float, Rows, Cols> A_oa = ans.matrixU() * ans.singularValues() * ans.matrixV().transpose();

    // Расчёт норм
    float norm_bdc = (A - A_bdc).norm();
    float norm_oa  = (A - A_oa).norm();

    // Вывод
    //fout << "Matrix size: " << Rows << "x" << Cols << "\n";
    //fout << "Norm (Eigen SVD): " << norm_bdc << "\n";
    //fout << "Norm (Ogita-Aishima): " << norm_oa << "\n";
    //fout << "Elapsed time (Ogita-Aishima): " << elapsed.count() << " sec\n";
    cout << "Matrix size: " << Rows << "x" << Cols << "\n";
    cout << "Norm (Eigen SVD): " << norm_bdc << "\n";
    cout << "Norm (Ogita-Aishima): " << norm_oa << "\n";
    cout << "Elapsed time (Ogita-Aishima): " << elapsed.count() << " sec\n";

    //fout.close();
}



int main() {
    //run_test<10, 10>();
    //run_test<20, 20>();
    //run_test<30, 30>();
    //run_test<40, 40>();
    //run_test<50, 50>();
    run_test<60, 60>();
    //run_test<50, 40>();
    //run_test<60, 40>();
    return 0;
}
