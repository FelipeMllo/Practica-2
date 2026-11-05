
#include "SimpleLinearRegression.h"
using namespace std;

SimpleLinearRegression::SimpleLinearRegression() : RegressionModel(1) {}

void SimpleLinearRegression::fit(double** X, double* y, int n) {
    double xm = 0.0, ym = 0.0;
    for (int i = 0; i < n; ++i) { xm += X[i][0]; ym += y[i]; }
    xm /= n; ym /= n;

    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = X[i][0] - xm;
        double dy = y[i]    - ym;
        num += dx * dy;
        den += dx * dx;
    }
    weights[0] = (den != 0.0) ? (num / den) : 0.0;
    bias       = ym - weights[0] * xm;
}

double SimpleLinearRegression::predict(const double* x) const {
    return bias + weights[0] * x[0];
}