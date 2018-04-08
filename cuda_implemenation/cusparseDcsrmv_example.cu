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

void read(string filename, double** P_sparse, int** row_ind, int** col_ind, int* nnz, int * n, int** nnzPerVectorA);

/********/
/* MAIN */
/********/
int main()
{
    // --- Initialize cuSPARSE
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

    /**************************/
    /* SETTING UP THE PROBLEM */
    /**************************/
    string filename = "data.dat";
	int *row_ind, *col_ind, *nnzPerVectorA;
	double* P_sparse;
	int nnz, n;
	read(filename, &P_sparse, &row_ind, &col_ind, &nnz, &n, &nnzPerVectorA);
    const int N     = n;                // --- Number of rows and columns

    // --- Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSafeCall(cusparseSetMatType     (descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));  

    int nnzA = nnz;                           // --- Number of nonzero elements in dense matrix A

    const int lda = N;                      // --- Leading dimension of dense matrix

    printf("Number of nonzero elements in dense matrix A = %i\n\n", nnzA);
    for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, nnzPerVectorA[i]);
    printf("\n");

    printf("\nOriginal matrix A in CSR format\n\n");
    for (int i = 0; i < nnzA; ++i) printf("A[%i] = %f ", i, P_sparse[i]); printf("\n");

    printf("\n");
    for (int i = 0; i < (N + 1); ++i) printf("P_sparse_RowIndices[%i] = %i \n", i, col_ind[i]); printf("\n");

    printf("\n");
    for (int i = 0; i < nnzA; ++i) printf("P_sparse_ColIndices[%i] = %i \n", i, row_ind[i]);  

	double *h_x_dense = (double*)malloc(N * sizeof(double));
	double *h_y_dense = (double*)malloc(N * sizeof(double));
	double *d_A; gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
	for (int k = 0; k < N; k++)
	{
        h_x_dense[k] = 1.;
        h_y_dense[k] = 0.;
	}
	double *d_x_dense;  gpuErrchk(cudaMalloc(&d_x_dense, N     * sizeof(double)));
	double *d_y_dense; gpuErrchk(cudaMalloc(&d_y_dense, N * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_x_dense, h_x_dense, N     * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_y_dense, h_y_dense, N * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A, P_sparse, nnzA * sizeof(*P_sparse), cudaMemcpyHostToDevice));
	
    printf("\n");
    for (int i = 0; i < N; ++i) printf("h_x[%i] = %f \n", i, h_x_dense[i]); printf("\n");

    const double alpha = 1.;
    const double beta  = 0.;
    cusparseSafeCall(cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, N, N, nnzA, &alpha, descrA, d_A, col_ind, row_ind, d_x_dense, 
                                    &beta, d_y_dense));
	cudaDeviceSynchronize();
	cout << "I am here" << endl;
    gpuErrchk(cudaMemcpy(h_y_dense,           d_y_dense,            N * sizeof(double), cudaMemcpyDeviceToHost));

    printf("\nResult vector\n\n");
    for (int i = 0; i < N; ++i) printf("h_y[%i] = %f ", i, h_y_dense[i]); printf("\n");

}
