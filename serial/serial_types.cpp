#include "serial_types.h"
#include <string>
#include <iostream>

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
