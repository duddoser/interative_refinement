#include "Ogita_Aishima.h"
#include <iostream>



int main() {
    using namespace Eigen;
    using namespace std;
    // Тест на матрице 4x3

    MatrixXf A = MatrixXf::Random(4, 3);

    cout << A << endl << endl;
    cout << "after OA algorithm" << "\n\n";

    Ogita_Aishima_SVD<float,4,3> Ans(A);

    cout << "Matrix U: \n" << Ans.matrixU() << endl;
    cout << "Matrix S: \n" << Ans.singularValues() << endl << endl;
    cout << "Matrix V: \n" << Ans.matrixV() << endl;
    cout << "Result: \n" << Ans.matrixU() * Ans.singularValues() * Ans.matrixV().transpose() << endl;


    return 0;
}