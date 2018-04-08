#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <cusparse_v2.h>
using namespace std;
#include "types.cuh"
#include <vector>

double* divide(double* x, double divisor, int n) {
    cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));
    const double alpha = 1/divisor;
    
    double* temp = new double[n];
    for (int i = 0; i < n; i++)
        temp[i] = 0; 
    
    double *d_zero;
    gpuErrchk(cudaMalloc(&d_zero, n * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_zero, temp, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    cublasSafeCall(cublasDaxpy(handle, n, &alpha, x,1,d_zero,1));
    gpuErrchk(cudaFree(x));
    gpuErrchk(cudaDeviceSynchronize());
    
    delete (temp);
	return d_zero;
}