#include "CSVReader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

static double parseNumber(const string& raw) {
    size_t a = raw.find_first_not_of(" \t\r\n");
    size_t b = raw.find_last_not_of(" \t\r\n");
    if (a == string::npos) return 0.0;
    string tok = raw.substr(a, b - a + 1);
    if (tok.empty()) return 0.0;
    try { return stod(tok); }
    catch (...) { return 0.0; }
}

void CSVReader::load(const string& filename, bool hasHeader,
                     double**& X, double*& y,
                     int& nSamples, int& nFeatures) {
    X = nullptr; y = nullptr; nSamples = 0; nFeatures = 0;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "[CSVReader] No se pudo abrir: " << filename << "\n";
        return;
    }

    vector<vector<double>> rows;
    string line;
    bool first = true;

    while (getline(file, line)) {
        size_t f = line.find_first_not_of(" \t\r\n");
        if (f == string::npos) continue;
        if (first && hasHeader) { first = false; continue; }
        first = false;

        stringstream ss(line);
        string tok;
        vector<double> row;
        while (getline(ss, tok, ','))
            row.push_back(parseNumber(tok));
        if (!row.empty()) rows.push_back(row);
    }
    file.close();

    nSamples  = (int)rows.size();
    nFeatures = (nSamples > 0) ? (int)rows[0].size() - 1 : 0;

    X = new double*[nSamples];
    y = new double[nSamples];
    for (int i = 0; i < nSamples; ++i) {
        X[i] = new double[nFeatures];
        for (int j = 0; j < nFeatures; ++j) X[i][j] = rows[i][j];
        y[i] = rows[i][nFeatures];
    }

    cout << "  [CSVReader] " << filename << " -> "
         << nSamples << " filas, " << nFeatures << " features\n";
    int k = (nSamples < 3) ? nSamples : 3;
    for (int i = 0; i < k; ++i) {
        cout << "    fila " << i << ": [";
        for (int j = 0; j < nFeatures; ++j)
            cout << X[i][j] << (j + 1 < nFeatures ? ", " : "");
        cout << "] -> y = " << y[i] << "\n";
    }
}

void CSVReader::freeData(double** X, double* y, int nSamples) {
    if (X) {
        for (int i = 0; i < nSamples; ++i) delete[] X[i];
        delete[] X;
    }
    delete[] y;
}