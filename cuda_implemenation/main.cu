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
const float EPS = 0.00001;
double alpha = 0.85;

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
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

struct flip_functor
{
  __host__ __device__
  void operator()(double &x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    if ( x > 0. ) x = 0;
    else x = 1;
  }
};

struct cust_functor
{

    const double a;
    const double b;
    const double alpha;

    cust_functor(double _a, double _b, double _alpha) : a(_a), b(_b), alpha(_alpha) {}

    __host__ __device__
    void operator()(double &x)
    {
        // note that using printf in a __device__ function requires
        // code compiled for a GPU with compute capability 2.0 or
        // higher (nvcc --arch=sm_20)
        x = alpha * (x + a) + (1 - alpha) * b;
    }
};

double* RunGPUPowerMethod(Matrix* P, double* x_new)
{
	printf("*************************************\n");
	double oldLambda = DBL_MAX;
	double lambda = 0;
	double alpha = 0.8;
    double dangling = 0.0;

	double* x = x_new;
    double* temp;
    double x_norm;

    double *d_ones, *h_ones;

    h_ones = new double[P->n];

    gpuErrchk(cudaMalloc(&d_ones, P->n * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_ones, h_ones, P->n * sizeof(double), cudaMemcpyHostToDevice));
    
    double *d_outnum;
    gpuErrchk(cudaMalloc(&d_outnum, P->n * sizeof(double)));
    MatrixMul(1.0, P, d_ones, d_outnum);
    
    // int dangling_test = thrust::count(d_th_outgoing, d_th_outgoing + P->n, 1);
    // int non_dangling_test = thrust::count(d_th_outgoing, d_th_outgoing + P->n, 0);
    thrust::device_ptr<double> d_x (x);
    thrust::device_ptr<double> d_th_outgoing (d_outnum);
    thrust::for_each(d_th_outgoing, d_th_outgoing + P->n, flip_functor());

    
    gpuErrchk(cudaMalloc(&x_new, P->n * sizeof(double)));
    //power loop
    cout << "Checkpoint" << endl;
	while(abs(lambda - oldLambda) > EPS)
	{
        dangling = thrust::inner_product(d_th_outgoing, d_th_outgoing + P->n, d_x, 0.0f);
        cout << "Dangling: " << dangling << endl;


        oldLambda = lambda;
        MatrixMul(1.0, P, x, x_new);

        thrust::device_ptr<double> d_th_x_new(x_new);
        thrust::for_each(d_th_x_new, d_th_x_new + P->n, cust_functor(dangling / P->n, ( (double) 1.0 )/P->n, alpha));
        
        x_norm = norm(x_new, P->n);
        x_new = divide (x_new, x_norm, P->n);

		temp = subtract(x, x_new, P->n);
		lambda = norm(temp, P->n);
		printf("GPU lamda: %f \n", lambda);
		x = x_new;
		x_new = temp;
	}
    printf("*************************************\n");
	return x;
}

double* UniformInit(int n) {
    double *x = new double[n];
    // cout << "random init: " << endl;
    for (int i = 0; i < n; i++) {
        // x[i] = (rand() % 100) / 100.0; 
        x[i] = 1.0 / (double)n; 
        // cout << x[i] << " ";
    }

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
    
    double* d_x = UniformInit(d_mat.n);
    
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

