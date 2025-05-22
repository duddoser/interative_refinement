// Реализация: Минаков Владислав
// КМБО-01-23 
// В этом коде реализуются тесты алгоримта Ogita_Aigsima
// с использованием точного матричного умножения

#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Sparse"
#include <chrono>
#include <random>
#include <fstream>


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

        matrix_mn Sigma = matrix_mn::Zero(m, n);
        Sigma.block(0, 0, n, n) = Sigma_n;

        matrix_nn R11 = R.topLeftCorner(n, n); // Верхний левый блок матрицы R размера n x n
        matrix_nn T1 = Tmn.topRows(n); // Квадратная матрица n x n из первых n строк матрицы T

        matrix_nn Calpha = T1 + Acc_Mul(R11, Sigma_n);
        matrix_nn Cbeta = T1.transpose() + Acc_Mul(S, Sigma_n);
        matrix_nn D = Acc_Mul(Sigma_n, Calpha) + Acc_Mul(Cbeta, Sigma_n);
        matrix_nn E = Acc_Mul(Calpha, Sigma_n) + Acc_Mul(Sigma_n, Cbeta);

        matrix_nn G = matrix_nn::Zero(n,n);
        matrix_mm F = matrix_mm::Zero(m, m);

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
            for (int j = n; j < m; j++)
                 F(i,j) = -Tmn(j,i)/Sigma_n(i,i);

        for (int i = n; i < m; i++)
            for (int j = 0; j < n; j++)
                F(i,j) = R(i,j)-F(j,i);

        for (int i = 0; i < n; i++)
        {
            G(i,i) = S(i,i) * 0.5;
            F(i,i) = R(i,i) * 0.5;
        }

        for (int i = n; i < m; ++i) 
            for (int j = n; j < m; ++j)
                F(i,j) = R(i,j) * 0.5;
            
        
        matrix_mm U = Ud + Acc_Mul(Ud, F);//Вычисление уточнённых значений левых сингулярных векторов
        matrix_nn V = Vd + Acc_Mul(Vd, G);//Вычисление уточнённых значений правых сингулярных векторов

        Ogita_Aishima_SVD<T,M,N> ANS;
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


using T = double;
using Mat = Matrix<T, Dynamic, Dynamic>;
int main() {

     vector<pair<int,int>> sizes             = {{10,10}};
                                             // Интервал сингулярных значений
     vector<pair<double,double>> intervals   = {{0,10},{0,100}};
                                             // Количество итераций
     vector<int> iterCounts                  = {1,5,10,20,50};
                                             // Уровень шума
     vector<T> noiseLevels                   = {1e-15,1e-10};


    ofstream file("test_mul.csv");
    file << "Size,Interval,CondNum,NoiseLevel,Iter,Rec_l1,Rec_l2,Time_ms,U_err,S_err,V_err\n";

    random_device rd; mt19937 gen(rd()); // Генератор случайных чисел

    for (auto [m,n] : sizes) {
        for (auto [a,b] : intervals) {
            // Генерация исходной матрицы A = U * S * V^T
            Mat U = Mat::Random(m,m);
            Mat V = Mat::Random(n,n);
            HouseholderQR<Mat> qrU(U), qrV(V);
            U = qrU.householderQ();
            V = qrV.householderQ();

            uniform_real_distribution<T> dist(a,b);
            vector<T> sv(n);
            for (int i = 0; i < n; ++i) sv[i] = dist(gen);
            sort(sv.begin(), sv.end(), greater<T>());

            Mat S = Mat::Zero(m,n);
            for (int i = 0; i < n; ++i) S(i,i) = sv[i];
            Mat A = U * S * V.transpose();
            T condNum = sv.front() / sv.back();

            for (T noiseLevel : noiseLevels) 
            {
                // Зашумление U и V
                Mat Uc = U;
                Mat Vc = V;
                uniform_real_distribution<T> noiseDist(0.0, noiseLevel);
                for (int i = 0; i < Uc.rows(); ++i)
                    for (int j = 0; j < Uc.cols(); ++j)
                        Uc(i,j) += noiseDist(gen);
                for (int i = 0; i < Vc.rows(); ++i)
                    for (int j = 0; j < Vc.cols(); ++j)
                        Vc(i,j) += noiseDist(gen);

                for (int it : iterCounts) {
                    Mat Un, Vn, Sn;
                    auto start = chrono::high_resolution_clock::now();
                    // Итеративное уточнение Ogita–Aishima SVD на основе шумных Uc, Vc
                    Ogita_Aishima_SVD<T, Dynamic, Dynamic> svd(A, Uc, Vc, it);
                    Un = svd.matrixU();
                    Sn = svd.singularValues();
                    Vn = svd.matrixV();
                    auto end = chrono::high_resolution_clock::now();


                    T r1 = (A - Un * Sn * Vn.transpose()).template lpNorm<1>();
                    T r2 = (A - Un * Sn * Vn.transpose()).norm();
                    T Ue = (Un - U).norm();
                    T Ve = (Vn - V).norm();
                    T Se = (Sn - S).norm();
                    double timeIt = chrono::duration<double, milli>(end - start).count();

                    file << m << "x" << n << ",[" << a << ";" << b << "],"
                        << condNum << "," << noiseLevel << "," << it << ","
                        << r1 << "," << r2 << "," << timeIt << ","
                        << Ue << "," << Se << "," << Ve << "\n";
                    cout << "Size: " << m << "x" << n
                         << " noise=" << noiseLevel << " it=" << it << " done " << timeIt << endl;

                }
                file << "\n";
            }
        }
    }

    file.close();
    cout << "Finished" << endl;
    return 0;
}
