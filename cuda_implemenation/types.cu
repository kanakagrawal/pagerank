#include "types.cuh"
#include "Utilities.cuh"
#include <cuda.h>
#include <string>

void read(std::string filename, double** P_sparse, int** row_ind, int** col_ind, int* nnz, int * n);

Matrix::Matrix(int n, int nnz, double *p, int *row_ind, int *col_ind, bool device) {
    this->n = n;
    this->nnz = nnz;
    this->p = p;
    this->row_ind = row_ind;
    this->col_ind = col_ind;
    this->device = device;
}

Matrix::Matrix(std::string filename) {
    read ( filename, &p, &row_ind, &col_ind, &nnz, &n);
    this->device = false;
}

Matrix Matrix::CopyToDevice() {
    double *d_p;
    int *d_row_ind;
    int *d_col_ind;

    gpuErrchk(cudaMalloc( &d_p, nnz * sizeof (double) ));
    gpuErrchk(cudaMalloc( &d_row_ind, nnz * sizeof (int) ));
    gpuErrchk(cudaMalloc( &d_col_ind, (n + 1) * sizeof (int) ));

    gpuErrchk(cudaMemcpy(d_p, p, nnz * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_ind, row_ind, nnz * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, col_ind, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    return Matrix ( n, nnz, d_p, d_row_ind, d_col_ind, true );
}

Matrix Matrix::CopyToHost() {
    double *h_p;
    int *h_row_ind;
    int *h_col_ind;

    h_p = new double[nnz];
    h_row_ind = new int[nnz];
    h_col_ind = new int[(n + 1)];

    gpuErrchk(cudaMemcpy(h_p, p, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_row_ind, row_ind, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_col_ind, col_ind, (n + 1) * sizeof(double), cudaMemcpyDeviceToHost));
    
    return Matrix ( n, nnz, h_p, h_row_ind, h_col_ind, false );
}

void Matrix::CopyToHost(Matrix* dest) {
    double *h_p;
    int *h_row_ind;
    int *h_col_ind;

     
    dest->clear();
    h_p = new double[nnz];
    h_row_ind = new int[nnz];
    h_col_ind = new int[(n + 1)];

    gpuErrchk(cudaMemcpy(h_p, p, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_row_ind, row_ind, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_col_ind, col_ind, (n + 1) * sizeof(double), cudaMemcpyDeviceToHost));

    dest->n = n;
    dest->nnz = nnz;
    dest->p = h_p;
    dest->row_ind = h_row_ind;
    dest->col_ind = h_col_ind;
    dest->device = false;
}

void Matrix::clear() {
    n = 0;
    nnz = 0;
    if (device) {
        gpuErrchk(cudaFree(p));
        gpuErrchk(cudaFree(col_ind));
        gpuErrchk(cudaFree(row_ind));
    }
    else{
        delete(p);
        delete(col_ind);
        delete(row_ind);
    }
}

Matrix::~Matrix() {
    this->clear ();
}