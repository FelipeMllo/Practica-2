#include "MultipleLinearRegression.h"
#include <cmath>
using namespace std;

MultipleLinearRegression::MultipleLinearRegression(int nFeatures)
    : RegressionModel(nFeatures) {}
void MultipleLinearRegression::fit(double** X, double* y, int n) {
    int p = nFeatures;
    int m = p + 1;

    double** A   = new double*[m];
    for (int i = 0; i < m; ++i) A[i] = new double[m]();
    double* rhs = new double[m]();
    double* theta = new double[m]();

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < m; ++i) {
            double xi = (i == 0) ? 1.0 : X[k][i - 1];
            rhs[i] += xi * y[k];
            for (int j = 0; j < m; ++j) {
                double xj = (j == 0) ? 1.0 : X[k][j - 1];
                A[i][j] += xi * xj;
            }
        }
    }

    for (int i = 0; i < m; ++i) theta[i] = rhs[i];

    for (int col = 0; col < m; ++col) {
        int piv = col;
        double best = fabs(A[col][col]);
        for (int r = col + 1; r < m; ++r) {
            if (fabs(A[r][col]) > best) { best = fabs(A[r][col]); piv = r; }
        }
        if (piv != col) {
            double* tR = A[col];    A[col]     = A[piv];     A[piv] = tR;
            double  tV = theta[col]; theta[col] = theta[piv]; theta[piv] = tV;
        }
        if (fabs(A[col][col]) < 1e-12) A[col][col] = 1e-8;

        for (int r = col + 1; r < m; ++r) {
            double f = A[r][col] / A[col][col];
            for (int c = col; c < m; ++c) A[r][c] -= f * A[col][c];
            theta[r] -= f * theta[col];
        }
    }

    for (int i = m - 1; i >= 0; --i) {
        double s = theta[i];
        for (int j = i + 1; j < m; ++j) s -= A[i][j] * theta[j];
        theta[i] = s / A[i][i];
    }

    bias = theta[0];
    for (int j = 0; j < p; ++j) weights[j] = theta[j + 1];

    for (int i = 0; i < m; ++i) delete[] A[i];
    delete[] A;
    delete[] rhs;
    delete[] theta;
}

double MultipleLinearRegression::predict(const double* x) const {
    double p = bias;
    for (int j = 0; j < nFeatures; ++j) p += weights[j] * x[j];
    return p;
}