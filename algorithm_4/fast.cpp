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

MatrixXd Acc_Mul(const MatrixXd& A, const MatrixXd& B, int k, double delta = 0.0) 
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


int main() {
    MatrixXd A = MatrixXd::Random(300, 300);
    MatrixXd B = MatrixXd::Random(300, 300);

    int k = 2;
    double delta = 0.2;

    auto start1 = chrono::high_resolution_clock::now();
    MatrixXd C1 = A * B;
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration1 = end1 - start1;

    auto start2 = chrono::high_resolution_clock::now();
    MatrixXd C2 = Acc_Mul(A, B, k, delta);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration2 = end2 - start2;

    cout << "Size :  300 x 300  "  << endl;

    cout << "Time for A * B:      " << duration1.count() << " ms" << endl;
    cout << "Time for Acc_Mul:   " << duration2.count() << " ms" << endl;

    cout << "Norm of result:  " << (C2 - C1).norm() << endl;

    return 0;
}