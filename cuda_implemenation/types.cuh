#ifndef TYPES_CUH
#define TYPES_CUH

#include <string>
class Matrix {
public:
    int n;
    int nnz;
    double *p;
    int *row_ind;
    int *col_ind;
    bool device;

    Matrix(int n, int nnz, double *p, int *row_ind, int *col_ind, bool device);
    Matrix(std::string filename);
    ~Matrix();
    Matrix CopyToDevice();
    Matrix CopyToHost();
    void CopyToHost(Matrix* dest);
    void clear();
};

#endif