#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "device_functions.h"
#include "types.cuh"
#include "Utilities.cuh"
#include <sys/time.h>
#include <string>
#include <float.h>
#include <iostream>
#include <fstream>
#include "find_top_k.cuh"

using namespace std;

// Variables to change
const float EPS = 0.000001;
double alpha = 0.85;

void MatrixMul(double alpha, Matrix *mat, double* d_x_dense, double *d_y_dense, cusparseHandle_t handle, cusparseMatDescr_t descrA);
// void MatrixMul(double alpha, Matrix *mat, double* x, double* x_new); // returns alpha * mat * x
double* subtract(double* d_x,double* d_y, int n);
double norm(double* d_x, int n);
double* divide(double* x, double divisor, int n);

void PrintArray(double* data, int n);
void DevicePrintArray(double* data, int n);
std::string ParseArguments( int argc, char **argv );

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>

struct add_functor
{

    const double a;

    add_functor(double _a) : a(_a) {}

    __host__ __device__
    void operator()(double &x)
    {
        x = x + a;
    }
};



double* RunGPUPowerMethod(Matrix* P, double* x_new)
{
	printf("*************************************\n");
	double oldLambda = DBL_MAX;
	double lambda = 0;

	double* x = x_new;
    double* temp;
    double x_norm, x_new_norm;
    double omega;
    int n = P->n;
        
    double *h_ones = new double[n]; 
    double *d_ones;
    for (int i = 0; i < n;i++) {
        h_ones[i] = 1.0;
    }

    gpuErrchk(cudaMalloc(&d_ones, n * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_ones, h_ones, n * sizeof(double), cudaMemcpyHostToDevice));

    
    gpuErrchk(cudaMalloc(&x_new, n * sizeof(double)));
    //power loop
    cublasHandle_t blas_handle;
    cublasSafeCall(cublasCreate(&blas_handle));

    // --- Initialize cuSPARSE
    cusparseHandle_t sparse_handle;    cusparseSafeCall(cusparseCreate(&sparse_handle));

    // --- Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSafeCall(cusparseSetMatType     (descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    cout << "Checkpoint" << endl;
    while(abs(lambda - oldLambda) > EPS)
    {
        oldLambda = lambda;
        MatrixMul(alpha, P, x, x_new, sparse_handle, descrA);

        x_norm = norm(x, n);
        x_new_norm = norm(x_new, n);
        omega = x_norm - x_new_norm;

        cout << "Omega: " << omega << endl;

        //   thrust::device_ptr<double> d_th_x_new(x_new);
        //   thrust::for_each(d_th_x_new, d_th_x_new + n, add_functor( omega / ( (double) n ) ));
        
        const double mult = omega / ( (double) n );
        cublasSafeCall(cublasDaxpy(blas_handle, n,&mult, d_ones, 1, x_new, 1));
        gpuErrchk(cudaDeviceSynchronize());

        temp = subtract(x, x_new, n);
        lambda = norm(temp, n);
        printf("CPU lamda: %f \n", lambda);
        cout << endl;
        x = x_new;
        x_new = temp;
    }
    printf("*************************************\n");
    return x;
}

double* UniformInit(int n) {
    double *x = new double[n];
    for (int i = 0; i < n; i++) {
        x[i] = 1.0 / n; 
    }
    cout << endl;
    double *d_x;
    gpuErrchk(cudaMalloc(&d_x, n * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    delete(x);
    return d_x;
}

int compare_pairs (const void * a, const void * b)
{
    pair<double, double> _a = *(pair<double, double>*)a;
    pair<double, double> _b = *(pair<double, double>*)b;

    return _a.second < _b.second; 

    // return ( ()->second - ((pair<double, double>*)b)->second );
}

int main(int argc, char** argv)
{
    float elapsed=0;
    cudaEvent_t start, stop;
    struct timeval t1, t2;
    double time;


    std::string filename;

    filename = ParseArguments(argc, argv);
    gettimeofday(&t1, 0);
    Matrix mat(filename);
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to read input:  %3.1f ms \n", time);


    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk( cudaEventRecord(start, 0));
    Matrix d_mat = mat.CopyToDevice();
    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize (stop) );
    gpuErrchk(cudaEventElapsedTime(&elapsed, start, stop) );
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    printf("The elapsed time in copying to device was %.2f ms\n", elapsed);


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


    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk( cudaEventRecord(start, 0));
    d_x = RunGPUPowerMethod(&d_mat, d_x);
    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize (stop) );
    gpuErrchk(cudaEventElapsedTime(&elapsed, start, stop) );
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    printf("The elapsed time in gpu was %.2f ms\n", elapsed);
    
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
    
    pair<double, double> *top_ten = new pair<double, double>[top];

    for (int i = 0; i < top; i++) {
        top_ten[i].first = ind[i];
        top_ten[i].second = x[ind[i]];
    }

    qsort (top_ten, top, sizeof(pair<double, double>), compare_pairs);
    
    cout << "Top " << top << " link IDs are: " << endl;
    for (int i = 0; i < top; i++) {
        cout << "ID: " << top_ten[i].first << " - " << top_ten[i].second << endl;
    }


    delete (ind);
    delete (top_ten);
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

