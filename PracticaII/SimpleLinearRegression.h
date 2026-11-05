
#ifndef SIMPLELINEARREGRESSION_H
#define SIMPLELINEARREGRESSION_H

#include "RegressionModel.h"
using namespace std;

class SimpleLinearRegression : public RegressionModel {
public:
    SimpleLinearRegression();
    void        fit(double** X, double* y, int n) override;
    double      predict(const double* x) const    override;
    const char* modelName() const override {
        return "SimpleLinearRegression (OLS cerrada)";
    }
};
#endif