#include <stdio.h>
#include <math.h>
#include "device_functions.h"
#include "types.cuh"
#include "Utilities.cuh"
#include <string>
#include <float.h>
#include <iostream>

using namespace std;

// Variables to change
const float EPS = 0.000001;
double alpha = 0.8;

void MatrixMul(double alpha, Matrix *mat, double* x, double* x_new); // returns alpha * mat * x
double* subtract(double* d_x,double* d_y, int n);
double norm(double* d_x, int n);

/*
void CPU_NormalizeW()
{
	int N = GlobalSize;
	float normW=0;
	for(int i=0;i<N;i++)
		normW += h_VecW[i] * h_VecW[i];
	
	normW = sqrt(normW);
	for(int i=0;i<N;i++)
		h_VecV[i] = h_VecW[i]/normW;
}
*/

void RunGPUPowerMethod(Matrix P, double* x_new)
{
	printf("*************************************\n");
	double oldLambda = DBL_MAX;
	double lambda = 0;
	double alpha = 0.8;

	double* x = new double[P.n];
    gpuErrchk(cudaMalloc(&x, P.n * sizeof(double)));
	//power loop
	while(abs(lambda - oldLambda) > EPS)
	{
		MatrixMul(alpha, &P, x, x_new);
		double* temp = new double[P.n];
		temp = subtract(x, x_new, P.n);
		lambda = norm(temp, P.n);
		printf("CPU lamda: %f \n", lambda);
		oldLambda = lambda;	
		x = x_new;
		x_new = temp;
	}
	printf("*************************************\n");
//	return x_new;
}

int main(int argc, char** argv)
{
    std::string filename("data.dat");
    Matrix mat(filename);
    Matrix d_mat = mat.CopyToDevice();
    double* d_x;
    gpuErrchk(cudaMalloc(&d_x, d_mat.n * sizeof(double)));
    RunGPUPowerMethod(d_mat, d_x);
    double *x = new double[d_mat.n];
    gpuErrchk(cudaMemcpy(x, d_x, d_mat.n * sizeof(double), cudaMemcpyDeviceToHost));
    for(int i = 0; i < d_mat.n; i++)
    {
    	cout << x[i] << " ";
    }
    cout << endl;
}

void PrintArray(float* data, int n)
{
    for (int i = 0; i < n; i++)
        printf("[%d] => %f\n",i,data[i]);
}

