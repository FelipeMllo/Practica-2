
#ifndef CSVREADER_H
#define CSVREADER_H

#include <string>
using namespace std;

class CSVReader {
public:
    static void load(const string& filename, bool hasHeader,
                     double**& X, double*& y,
                     int& nSamples, int& nFeatures);
    static void freeData(double** X, double* y, int nSamples);
};
#endif