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

vector<double> subtract(vector<double> x,vector<double> y)
{
	if (x.size() != y.size()){
		cout<<"Size mismatch for array substraction"<<endl;
		exit(1);
	}

	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));
	const double alpha = -1.0;

	double *d_x; gpuErrchk(cudaMalloc(&d_x, x.size() * sizeof(*d_x)));
	double *d_y; gpuErrchk(cudaMalloc(&d_y, y.size() * sizeof(*d_y)));
	gpuErrchk(cudaMemcpy(d_x, x.data(), x.size() * sizeof(*d_x), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_y, y.data(), y.size() * sizeof(*d_y), cudaMemcpyHostToDevice));

	cublasSafeCall(cublasDaxpy(handle, x.size(),&alpha, d_y,1,d_x,1));
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(x.data(),d_x, x.size() * sizeof(*d_x), cudaMemcpyDeviceToHost));
	return x;
}



void sub_test(){
	vector<double>x(7,300);	
	vector<double>y(7,200);
	cout<<subtract(x,y)[6]<<endl;
}
