// Шаги тестирования:
// 1. Задаётся набор размеров матриц m×n для проверки разных форм и размерностей.
// 2. Для каждого размера генерируются случайные Ортогональные матрицы U и V.
// 3. Определяются сингулярные значения из равномерного распределения на интервалах [0,1], [0,10], [0,100], [0,1000].
// 4. Собирается тестовая матрица A = U * S * V^T и вычисляется её число обусловленности.
// 5. Создаются зашумлённые начальные приближения Uc и Vc путём добавления равномерного шума различных уровней (1e-20, 1e-15, 1e-10, 1e-5).
// 6. Для каждого уровня шума и числа итераций выполняется:
//    - Запуск конструктора Ogita_Aishima_SVD(A, Uc, Vc, iterations) для уточнения U, S, V.
//    - Замер времени работы уточнения.
//    - Вычисление метрик: L1- и L2- нормы ошибки восстановления A, нормы ошибок U, S, V.
// 7. Все результаты (размер матрицы, интервал сингулярных значений, число обусловленности, уровень шума, итерации, метрики и время) записываются в CSV-файл.

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <limits>
#include <algorithm>
#include "Ogita_Aishima.h"

using namespace std;
using namespace Eigen;
using T = long double;
using Mat = Matrix<T, Dynamic, Dynamic>;

int main() {
    //                                         // Размеры матриц
    // vector<pair<int,int>> sizes             = {{3,3},{5,4},{7,7},{10,10},{15,10},{15,15},{20,20},{20,17}, {30,30}, {30,25}, {52,38}, {50,50}};
    //                                         // Интервал сингулярных значений
    // vector<pair<double,double>> intervals   = {{0,1},{0,10},{0,100},{0,1000}};
    //                                         // Количество итераций
    // vector<int> iterCounts                  = {1,5,10,20,30,50,100};
    //                                         // Уровень шума
    // vector<T> noiseLevels                   = {1e-20L,1e-15L,1e-10L,1e-5L};

    // ---- Big matrix
                                            // Размеры матриц
    vector<pair<int,int>> sizes             = {{1000,1000}};
                                            // Интервал сингулярных значений
    vector<pair<double,double>> intervals   = {{0,100}};
                                            // Количество итераций
    vector<int> iterCounts                  = {5};
                                            // Уровень шума
    vector<T> noiseLevels                   = {1e-15L};


    ofstream file("tests_on_big_matrix.csv");
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

            uniform_real_distribution<long double> dist(a,b);
            vector<T> sv(n);
            for (int i = 0; i < n; ++i) sv[i] = dist(gen);
            sort(sv.begin(), sv.end(), greater<T>());

            Mat S = Mat::Zero(m,n);
            for (int i = 0; i < n; ++i) S(i,i) = sv[i];
            Mat A = U * S * V.transpose();
            long double condNum = sv.front() / sv.back();

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


                    long double r1 = (A - Un * Sn * Vn.transpose()).template lpNorm<1>();
                    long double r2 = (A - Un * Sn * Vn.transpose()).norm();
                    long double Ue = (Un - U).norm();
                    long double Ve = (Vn - V).norm();
                    long double Se = (Sn - S).norm();
                    double timeIt = chrono::duration<double, milli>(start - end).count();

                    file << m << "x" << n << ",[" << a << ";" << b << "],"
                        << condNum << "," << noiseLevel << "," << it << ","
                        << r1 << "," << r2 << "," << timeIt << ","
                        << Ue << "," << Se << "," << Ve << "\n";
                    cout << "Size: " << m << "x" << n
                         << " noise=" << noiseLevel << " it=" << it << " done\n";

                    // Сохранение матриц
                    ofstream fa("A_mat.txt");
                    ofstream fu("U_mat.txt");
                    ofstream fs("S_mat.txt");
                    ofstream fv("V_mat.txt");

                    ofstream fa_res("A_res.txt");
                    ofstream fu_res("U_res.txt");
                    ofstream fs_res("S_res.txt");
                    ofstream fv_res("V_res.txt");

                    constexpr int prec = std::numeric_limits<long double>::max_digits10;
                    fa << setprecision(prec);
                    fs << setprecision(prec);
                    fu << setprecision(prec);
                    fv << setprecision(prec);
                    fa_res << setprecision(prec);
                    fs_res << setprecision(prec);
                    fu_res << setprecision(prec);
                    fv_res << setprecision(prec);

                    fa << A << "\n";
                    fu << U << "\n";
                    fs << S << "\n";
                    fv << V << "\n";

                    Mat A_rec = Un * Sn * Vn.transpose();

                    fa_res << A_rec << "\n";
                    fu_res << Un << "\n";
                    fs_res << Sn << "\n";
                    fv_res << Vn << "\n";

                    fa.close(); fu.close(); fs.close(); fv.close();
                    fa_res.close(); fu_res.close(); fs_res.close(); fv_res.close();
                }
                file << "\n";
            }
        }
    }

    file.close();
    cout << "Finished" << endl;
    return 0;
}
