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

int norm(vector<double> x)
{
	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));
	
	double *d_x; gpuErrchk(cudaMalloc(&d_x, x.size() * sizeof(*d_x)));
	gpuErrchk(cudaMemcpy(d_x, x.data(), x.size() * sizeof(*d_x), cudaMemcpyHostToDevice));			
	double answer;
	cublasSafeCall(cublasDasum(handle, x.size(),d_x,1,&answer));
	return answer;
}

int main(){
	vector<double>x(7,300);	
	cout<<norm(x)<<endl;
}
