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
#include "find_top_k.cuh"

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



double* RunGPUPowerMethod(Matrix* P, double* x_new)
{
	printf("*************************************\n");
	double oldLambda = DBL_MAX;
	double lambda = 0;
	double alpha = 0.8;

	double* x = x_new;
    double* temp;
    double x_norm;
    gpuErrchk(cudaMalloc(&x_new, P->n * sizeof(double)));
    //power loop
    cout << "Checkpoint" << endl;
	while(abs(lambda - oldLambda) > EPS)
	{
		oldLambda = lambda;
        MatrixMul(alpha, P, x, x_new);
        DevicePrintArray(x,P->n);
        DevicePrintArray(x_new,P->n);
        x_norm = norm(x_new, P->n);
        x_new = divide (x_new, x_norm, P->n);

		temp = subtract(x, x_new, P->n);
		lambda = norm(temp, P->n);
		printf("CPU lamda: %f \n", lambda);
		x = x_new;
		x_new = temp;
	}
    printf("*************************************\n");
	return x;
}

double* RandomInit(int n) {
    double *x = new double[n];
    // cout << "random init: " << endl;
    srand(0);
    for (int i = 0; i < n; i++) {
        x[i] = (rand() % 100) / 100.0; 
        // cout << x[i] << " ";
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
	// mat.print();
    Matrix d_mat = mat.CopyToDevice();
    
    double* d_x = RandomInit(d_mat.n);
    // Normalizing the vector d_x
    double x_norm = norm(d_x, d_mat.n);
    d_x = divide (d_x, x_norm, d_mat.n);

#ifdef FDEBUG
    mat.print();
#endif
    
#ifdef DEBUG
    Matrix temp = d_mat.CopyToHost();
    cout << temp.n << " - " << temp.nnz << endl; 
    cout << mat.n << " - " << mat.nnz << endl; 
    cout << "# Start 10 Elems" << endl;
    for (int i = 0; i < 10; i++){
        cout << temp.p[i] << " " << temp.row_ind[i] << " " << temp.col_ind[i] << endl; 
        cout << mat.p[i] << " " << mat.row_ind[i] << " " << mat.col_ind[i] << endl; 
    }
        
    cout << "# End 10 Elems" << endl;
    for (int i = 0; i < 10; i++){
        cout << temp.p[temp.nnz - i] << " " << temp.row_ind[temp.nnz - i] << " " << temp.col_ind[temp.n - i] << endl; 
        cout << mat.p[temp.nnz - i] << " " << mat.row_ind[temp.nnz - i] << " " << mat.col_ind[temp.n - i] << endl; 
    }
#endif

    d_x = RunGPUPowerMethod(&d_mat, d_x);
    
    double *x = new double[d_mat.n];
    gpuErrchk(cudaMemcpy(x, d_x, d_mat.n * sizeof(double), cudaMemcpyDeviceToHost));

    ofstream f("output.out");    
    for(int i = 0; i < d_mat.n; i++)
    {
        f << i + 1 << " " << x[i] << endl;
    }
    f.close();
    
    int top = 10 < mat.n ? 10 : mat.n;
    size_t *ind = new size_t[top];
    kthLargest(x, mat.n, top, ind);

    cout << "Top " << top << " link IDs are: " << endl;
    for (int i = 0; i < top; i++) {
        cout << "ID: " << ind[i] << " - " << x[ind[i]] << endl;
    }
    delete (ind);
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

