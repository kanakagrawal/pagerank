#include "types.cuh"
#include <string>
void read(std::string filename, double** P_sparse, int** row_ind, int** col_ind, int* nnz, int * n);

Matrix::Matrix(std::string filename) {
    read ( filename, &p, &row_ind, &col_ind, &nnz, &n);
}

Matrix Matrix::CopyToDevice() {

}