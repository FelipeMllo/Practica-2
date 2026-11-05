
#include "RegressionModel.h"
#include <cmath>
using namespace std;

void RegressionModel::allocate(int n) {
    nFeatures = n;
    weights   = new double[n]();
    mean      = new double[n]();
    stddev    = new double[n]();
    for (int i = 0; i < n; ++i) stddev[i] = 1.0;
    bias   = 0.0;
    scaled = false;
}

void RegressionModel::deallocate() {
    delete[] weights; weights = nullptr;
    delete[] mean;    mean    = nullptr;
    delete[] stddev;  stddev  = nullptr;
}

void RegressionModel::copyFrom(const RegressionModel& o) {
    nFeatures = o.nFeatures;
    bias      = o.bias;
    scaled    = o.scaled;
    weights = new double[nFeatures];
    mean    = new double[nFeatures];
    stddev  = new double[nFeatures];
    for (int i = 0; i < nFeatures; ++i) {
        weights[i] = o.weights[i];
        mean[i]    = o.mean[i];
        stddev[i]  = o.stddev[i];
    }
}

RegressionModel::RegressionModel(int n) { allocate(n); }
RegressionModel::RegressionModel(const RegressionModel& o) { copyFrom(o); }

RegressionModel& RegressionModel::operator=(const RegressionModel& o) {
    if (this != &o) { deallocate(); copyFrom(o); }
    return *this;
}

RegressionModel::~RegressionModel() { deallocate(); }

double RegressionModel::score(double** X, double* y, int n) const {
    double meanY = 0.0;
    for (int i = 0; i < n; ++i) meanY += y[i];
    meanY /= n;

    double ssRes = 0.0, ssTot = 0.0;
    for (int i = 0; i < n; ++i) {
        double p = predict(X[i]);
        ssRes += (y[i] - p) * (y[i] - p);
        ssTot += (y[i] - meanY) * (y[i] - meanY);
    }
    double mse = ssRes / n;
    double r2  = (ssTot > 0.0) ? (1.0 - ssRes / ssTot) : 0.0;
    cout << "  MSE = " << mse << "\n  R2  = " << r2  << "\n";
    return mse;
}

void RegressionModel::data_scaling(double** X, int n) {
    for (int j = 0; j < nFeatures; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += X[i][j];
        mean[j] = s / n;

        double v = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = X[i][j] - mean[j];
            v += d * d;
        }
        stddev[j] = sqrt(v / n);
        if (stddev[j] == 0.0) stddev[j] = 1.0;
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < nFeatures; ++j)
            X[i][j] = (X[i][j] - mean[j]) / stddev[j];
    scaled = true;
}

double** RegressionModel::applyScaling(double** X, int n) const {
    double** out = new double*[n];
    for (int i = 0; i < n; ++i) {
        out[i] = new double[nFeatures];
        for (int j = 0; j < nFeatures; ++j)
            out[i][j] = (X[i][j] - mean[j]) / stddev[j];
    }
    return out;
}

void RegressionModel::print() const {
    cout << "  bias = " << bias << "\n";
    for (int j = 0; j < nFeatures; ++j)
        cout << "  w[" << j << "] = " << weights[j] << "\n";
}