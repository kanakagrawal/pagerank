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

double* subtract(double* d_x,double* d_y, int n)
{

	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));
	const double alpha = -1.0;
	cublasSafeCall(cublasDaxpy(handle, n,&alpha, d_y,1,d_x,1));
	gpuErrchk(cudaDeviceSynchronize());
	return d_x;
}


/*
void sub_test(){
	vector<double>x(7,300);	
	vector<double>y(7,200);
	cout<<subtract(x,y)[6]<<endl;
}
*/
