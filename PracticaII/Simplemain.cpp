#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>
#include "RegressionModel.h"
#include "SimpleLinearRegression.h"
#include "CSVReader.h"
using namespace std;

SimpleLinearRegression globalModel;

void runSimple(const string& path) {

    double** X = nullptr; double* y = nullptr;
    int n = 0, f = 0;
    CSVReader::load(path, true, X, y, n, f);
    if (!X) return;

    if (f != 1)
        cout << f;

    int nTrain = (int)(0.8 * n);
    int nTest  = n - nTrain;

    int* idx = new int[n];
    for (int i = 0; i < n; ++i) idx[i] = i;
    mt19937 rng(42);
    shuffle(idx, idx + n, rng);

    double** Xs = new double*[n];
    double*  ys = new double[n];
    for (int i = 0; i < n; ++i) {
        Xs[i] = X[idx[i]];
        ys[i] = y[idx[i]];
    }

    SimpleLinearRegression slr;
    slr.fit(Xs, ys, nTrain);

    cout << "\n pesos aprendidosn";
    slr.print();
    cout << "\n errores en TRAIN \n";
    slr.score(Xs, ys, nTrain);
    cout << "errores en TEST\n";
    slr.score(Xs + nTrain, ys + nTrain, nTest);
    cout << "  predict(X_test[0]) = " << slr.predict(Xs[nTrain])
         << "  (y real = " << ys[nTrain] << ")\n";

    delete[] idx;
    delete[] Xs;
    delete[] ys;

    CSVReader::freeData(X, y, n);
}

void localScopeDemo() {
    SimpleLinearRegression localModel;
    cout << "  [Local]  &localModel = " << &localModel
         << "  | sizeof = " << sizeof(localModel) << " B\n"
         << "           weights ptr = " << localModel.getWeights() << "\n";
}

void memoryAnalysis() {

    cout << "sizeof(SimpleLinearRegression) = "
         << sizeof(SimpleLinearRegression) << " B\n\n";

    cout << "[Global]  &globalModel = " << &globalModel
         << "  | weights ptr = " << globalModel.getWeights() << "\n";

    localScopeDemo();

    SimpleLinearRegression* dyn = new SimpleLinearRegression();
    cout << "[Dinamica]    dyn         = " << dyn
         << "  | weights ptr = " << dyn->getWeights() << "\n";
    delete dyn;

    const int N = 5;
    SimpleLinearRegression* arr = new SimpleLinearRegression[N];
    cout << "\n[VECTOR DINAMICO] new SimpleLinearRegression[" << N << "]\n"
         << "  N * sizeof(SLR) = " << N * sizeof(SimpleLinearRegression)
         << " bytes\n"
         << "  Address of arr[0] = " << &arr[0] << "\n";
    for (int i = 0; i < N; ++i)
        cout << "    arr[" << i << "] @ " << &arr[i]
             << " -> weights = " << arr[i].getWeights() << "\n";
    delete[] arr;

}

int main(int argc, char** argv) {
    cout << fixed << setprecision(4);

    string path;
    if (argc >= 2) {
        path = argv[1];
    } else {
        path = "Ice_cream_selling_data.csv";
    }

    runSimple(path);
    memoryAnalysis();
    return 0;
}