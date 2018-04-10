#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <string>
#include <fstream>

#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cusparse_v2.h>
using namespace std;
#include "types.cuh"
#include <vector>

void read(string filename, double** P_sparse, int** row_ind, int** col_ind, int* nnz, int * n);

// returns alpha * mat * x
void MatrixMul(double alpha, Matrix *mat, double* d_x_dense, double *d_y_dense, cusparseHandle_t handle, cusparseMatDescr_t descrA)
{
    

    const int N = mat->n;                // --- Number of rows and columns
    int nnzA = mat->nnz;                           // --- Number of nonzero elements in dense matrix A  

#ifdef DEBUG
    printf("\nOriginal matrix A in CSR format\n\n");

    printf("\n");
    for (int i = 0; i < nnzA; ++i) printf("P_sparse_ColIndices\n");  
#endif

	double *h_y_dense = (double*)malloc(N * sizeof(double));
	for (int k = 0; k < N; k++)
	{
        h_y_dense[k] = 0.;
    }
	gpuErrchk(cudaMemcpy(d_y_dense, h_y_dense, N * sizeof(double), cudaMemcpyHostToDevice));
    // Matrix mat = mat->CopyToDevice();
    free(h_y_dense);

    
    const double beta  = 0.;
    cusparseSafeCall(cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, N, N, nnzA, &alpha, descrA, mat->p, mat->col_ind, mat->row_ind, d_x_dense, 
                                    &beta, d_y_dense));
	gpuErrchk(cudaDeviceSynchronize()); 
#ifdef DEBUG
    gpuErrchk(cudaMemcpy(h_y_dense, d_y_dense, N * sizeof(double), cudaMemcpyDeviceToHost));
    printf("\nResult vector\n\n");
    for (int i = 0; i < N; ++i) printf("h_y[%i] = %f ", i, h_y_dense[i]); printf("\n");
#endif
}

// testing matmul
void mul_test () {
    string filename("data.dat");
    Matrix mat (filename);
    
    // vector<double> x = {1.0, 1.0, 1.0, 1.0};
    // vector<double> y (MatrixMul(1.0, &mat, x));

    // for (int i = 0; i < y.size(); i++)
        // cout << y[i] << endl;
}
