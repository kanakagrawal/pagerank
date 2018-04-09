#include "serial_types.h"
#include <string>
#include <iostream>
#include <vector>

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

void Matrix::clear() {
    n = 0;
    nnz = 0;
    if (device) {
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

void Matrix::print() {
    if (device) {
    }
    else {  
        std::cout << n << " " << nnz << std::endl;
        std::cout<<"col_ind"<<std::endl;	  
        for (int i = 0; i < (n+1); i++) {
            std::cout << col_ind[i] << " ";
        }
        std::cout << std::endl;
        std::cout<<"row_ind"<<std::endl;	  
        for (int i = 0; i < nnz; i++) {
            std::cout << row_ind[i] << " ";
        }
        std::cout << std::endl;   
    }
} 

// https://raw.githubusercontent.com/karlrupp/blog-sparse-matrix-transpose/master/from_csr.hpp
Matrix* Matrix:: transpose(){
    int* B_rows = new int[n+1];
    int* B_cols = new int[nnz];
    double* B_values = new double[nnz];
    for(int i = 0;i<=n;i++){
        B_rows[i] = 0;
    }
    for(int i = 0;i<nnz;i++){
        B_cols[i] = 0;
        B_values[i] = 0;
    }
    int row_start = 0;
    for (int row = 0; row < n; ++row){
        int row_stop  = col_ind[row+1];
        for (int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
            B_rows[row_ind[nnz_index]] += 1;
        row_start = row_stop;
    }

      // Bring row-start array in place using exclusive-scan:
    int offset = 0;
    for (std::size_t row = 0; row < n; ++row)
    {
        int tmp = B_rows[row];
        B_rows[row] = offset;
        offset += tmp;
    }
    B_rows[n] = offset;

    std::vector<int> B_offsets(n+1); // index of first unwritten element per row
    for(int i = 0;i<=n;i++){
        B_offsets[i] = B_rows[i];
    }
    row_start = col_ind[0];
    for (std::size_t row = 0; row < n; ++row)
    {
        int row_stop  = col_ind[row+1];

        for (int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
        {
            int col_in_A = row_ind[nnz_index];
            int B_nnz_index = B_offsets[col_in_A];
            B_cols[B_nnz_index] = row;
            B_values[B_nnz_index] = p[nnz_index];
            B_offsets[col_in_A] += 1;
        }
        row_start = row_stop;
    }
    Matrix* B;
    B = new Matrix(n,nnz,B_values,B_cols,B_rows,0);
    return B;
}