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

    Matrix(std::string filename);
    Matrix CopyToDevice();
    Matrix CopyToHost();
};

#endif