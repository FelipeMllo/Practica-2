
#ifndef MULTIPLELINEARREGRESSION_H
#define MULTIPLELINEARREGRESSION_H

#include "RegressionModel.h"

class MultipleLinearRegression : public RegressionModel {
public:

    MultipleLinearRegression(int nFeatures = 4);

    void        fit(double** X, double* y, int n) override;
    double      predict(const double* x) const    override;

    const char* modelName() const override {
        return "MultipleLinearRegression (OLS - Ecuacion Normal)";
    }
};
#endif