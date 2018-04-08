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

int norm(double* d_x, int n)
{
	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));
	double answer;
	cublasSafeCall(cublasDasum(handle, n, d_x, 1, &answer));
	gpuErrchk(cudaDeviceSynchronize());	
	return answer;
}

void norm_test(){
	vector<double>x(7,300);	
	cout<<norm(x)<<endl;
}
