
#ifndef REGRESSIONMODEL_H
#define REGRESSIONMODEL_H

#include <iostream>
using namespace std;

class RegressionModel {
protected:
    double* weights;
    double  bias;
    int     nFeatures;
    double* mean;
    double* stddev;
    bool    scaled;

    void allocate(int n);
    void deallocate();
    void copyFrom(const RegressionModel& o);

public:
    RegressionModel(int n);
    RegressionModel(const RegressionModel& o);
    RegressionModel& operator=(const RegressionModel& o);
    virtual ~RegressionModel();

    virtual void   fit(double** X, double* y, int n) = 0;
    virtual double predict(const double* x) const    = 0;

    virtual double   score(double** X, double* y, int n) const;
    virtual void     data_scaling(double** X, int n);
    virtual double** applyScaling(double** X, int n) const;

    const double* getWeights()     const { return weights; }
    double        getBias()        const { return bias; }
    int           getNumFeatures() const { return nFeatures; }
    bool          isScaled()       const { return scaled; }

    virtual void        print()     const;
    virtual const char* modelName() const = 0;
};
#endif