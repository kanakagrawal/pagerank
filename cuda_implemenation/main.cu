#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "device_functions.h"
#include "types.cuh"
#include "Utilities.cuh"
#include <string>
#include <float.h>
#include <iostream>
#include <fstream>

using namespace std;

// Variables to change
const float EPS = 0.000001;
double alpha = 0.8;

void MatrixMul(double alpha, Matrix *mat, double* x, double* x_new); // returns alpha * mat * x
double* subtract(double* d_x,double* d_y, int n);
double norm(double* d_x, int n);
double* divide(double* x, double divisor, int n);

void PrintArray(double* data, int n);
void DevicePrintArray(double* data, int n);
std::string ParseArguments( int argc, char **argv );

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



double* RunGPUPowerMethod(Matrix P, double* x_new)
{
	printf("*************************************\n");
	double oldLambda = DBL_MAX;
	double lambda = 0;
	double alpha = 0.8;

	double* x = x_new;
    double* temp;
    double x_norm;
    gpuErrchk(cudaMalloc(&x_new, P.n * sizeof(double)));
	//power loop
	while(abs(lambda - oldLambda) > EPS)
	{
		oldLambda = lambda;
        MatrixMul(alpha, &P, x, x_new);
        x_norm = norm(x_new, P.n);
        x_new = divide (x_new, x_norm, P.n);

		temp = subtract(x, x_new, P.n);
		lambda = norm(temp, P.n);
		printf("CPU lamda: %f \n", lambda);
		x = x_new;
		x_new = temp;
	}
    printf("*************************************\n");
	return x;
}

double* RandomInit(int n) {
    double *x = new double[n];
    cout << "random init: " << endl;
    for (int i = 0; i < n; i++) {
        x[i] = (rand() % 100) / 100.0; 
        cout << x[i] << " ";
    }
    cout << endl;
    double *d_x;
    gpuErrchk(cudaMalloc(&d_x, n * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    delete(x);
    return d_x;
}

int main(int argc, char** argv)
{
    std::string filename;
    filename = ParseArguments(argc, argv);

    Matrix mat(filename);
    mat.print();
    Matrix d_mat = mat.CopyToDevice();
    d_mat.print();
    
    double* d_x = RandomInit(d_mat.n);

    d_x = RunGPUPowerMethod(d_mat, d_x);
    
    double *x = new double[d_mat.n];
    gpuErrchk(cudaMemcpy(x, d_x, d_mat.n * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());
    cout << "Adfa" << endl;
    double max_val = x[0];
    int max_pos = 0; 

    ofstream f("output.out");
    
    for(int i = 0; i < d_mat.n; i++)
    {
        cout << x[i] << " ";
        if(max_val < x[i]) {
            max_pos = i;
            max_val = x[i];
        }
        f << i + 1 << " " << x[i] << endl;
    }
    cout << endl;
    cout << "Max Link ID: " << max_pos + 1 <<  " with PR " << max_val << endl;
    f.close();
}

void PrintArray(double* data, int n)
{
    printf("array ~~~~~>\n");
    for (int i = 0; i < n; i++)
        printf("[%d] => %lf\n",i,data[i]);
}

void DevicePrintArray(double* data, int n) {
    double *temp = new double[n];
    gpuErrchk(cudaMemcpy(temp, data, n * sizeof(double), cudaMemcpyDeviceToHost));    
    PrintArray(temp, n);
    delete(temp);
}

std::string ParseArguments( int argc, char **argv ) {
    if (argc == 2) {
        return string(argv[1]);
    }
    else {
        return string("data.dat");
    }
}

