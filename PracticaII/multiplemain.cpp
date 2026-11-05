#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>
#include "RegressionModel.h"
#include "MultipleLinearRegression.h"
#include "CSVReader.h"
using namespace std;

MultipleLinearRegression globalModel(4);


void runMultiple(const string& path) {

    double** X = nullptr; double* y = nullptr;
    int n = 0, f = 0;
    CSVReader::load(path, true, X, y, n, f);
    if (!X) return;

    int nTrain = (int)(0.8 * n);
    int nTest  = n - nTrain;

    int* idx = new int[n];
    for (int i = 0; i < n; ++i) idx[i] = i;
    mt19937 rng(42);
    shuffle(idx, idx + n, rng);

    double** Xs = new double*[n];
    double*  ys = new double[n];
    for (int i = 0; i < n; ++i) { Xs[i] = X[idx[i]]; ys[i] = y[idx[i]]; }

    MultipleLinearRegression mlr(f);
    mlr.fit(Xs, ys, nTrain);

    cout << "\n pesos aprendidos (unidades originales)";
    mlr.print();
    cout << "\n errores en TRAIN\n";
    mlr.score(Xs, ys, nTrain);
    cout << " errores en TEST\n";
    mlr.score(Xs + nTrain, ys + nTrain, nTest);
    cout << "  predict(X_test[0]) = " << mlr.predict(Xs[nTrain])
         << "  (y real = " << ys[nTrain] << ")\n";

    delete[] idx;
    delete[] Xs;
    delete[] ys;
    CSVReader::freeData(X, y, n);
}

void memoryAnalysisMultiple() {

    cout << "sizeof(MultipleLinearRegression) = "
         << sizeof(MultipleLinearRegression) << " B\n\n";

    cout << "[GLOBAL]   &globalModel = " << &globalModel
         << " | weights ptr = " << globalModel.getWeights() << "\n"
         << "           (segmento .data/.bss)\n";

    MultipleLinearRegression localModel(4);
    cout << "[LOCAL]    &localModel  = " << &localModel
         << " | weights ptr = " << localModel.getWeights() << "\n"
         << "           (pila / stack)\n";

    MultipleLinearRegression* dyn = new MultipleLinearRegression(4);
    cout << "[DINAMICO] dyn         = " << dyn
         << " | weights ptr = " << dyn->getWeights() << "\n"
         << "           (heap)\n";
    delete dyn;

    const int N = 5;
    MultipleLinearRegression* arr = new MultipleLinearRegression[N];
    cout << "\n[VECTOR DINAMICO] new MultipleLinearRegression[" << N << "]\n"
         << "  N * sizeof(MLR) = " << N << " * " << sizeof(MultipleLinearRegression)
         << " = " << N * sizeof(MultipleLinearRegression) << " bytes\n"
         << "  Address of arr[0] = " << &arr[0] << "\n"
         << "  (bloque CONTIGUO de N objetos en heap + array cookie)\n";
    for (int i = 0; i < N; ++i)
        cout << "    arr[" << i << "] @ " << &arr[i]
             << " -> weights = " << arr[i].getWeights() << "\n";
    delete[] arr;
}

int main(int argc, char** argv) {
    cout << fixed << setprecision(4);
    string path = (argc >= 2) ? argv[1] : "student_exam_scores.csv";

    runMultiple(path);
    memoryAnalysisMultiple();
    return 0;
}