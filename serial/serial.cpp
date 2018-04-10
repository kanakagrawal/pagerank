#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <float.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>
#include "types.h"
#include "find_top_k.h"

using namespace std;

const float EPS = 0.000001;
double alpha = 0.85;

void SerialMatrixMul(double alpha, Matrix *mat, double* x, double* x_new); // returns alpha * mat * x

void PrintArray(double* data, int n);
std::string ParseArguments( int argc, char **argv );

double* RunCPUPowerMethod(Matrix* P, double* x_new)
{
	printf("*************************************\n");
	double oldLambda = DBL_MAX;
	double lambda = 0;
	double alpha = 0.8;

	double* x = x_new;
    double* temp;
    double omega;
    double x_norm, x_new_norm;
    x_new = new double[P->n];
    cout << "Checkpoint" << endl;
    while(abs(lambda - oldLambda) > EPS)
    {
      oldLambda = lambda;
      SerialMatrixMul(alpha, P, x, x_new);
      x_norm = 0;
      for(int i =0; i<P->n; i++){
        x_norm += abs(x[i]);
    }
    x_new_norm = 0;
    for(int i =0; i<P->n; i++){
        x_new_norm += abs(x_new[i]);
    }
    omega = x_norm - x_new_norm;

    for(int i =0; i<P->n; i++){
        x_new[i] += omega/P->n;
    }
    lambda = 0;
    for(int i = 0; i<P->n; i++){
        lambda += abs(x[i] - x_new[i]);
    }
    printf("CPU lamda: %f \n", lambda);
    double* temp = x;
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
    return x;
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
    struct timeval t1, t2;
    double time;
    std::string filename;
    filename = ParseArguments(argc, argv);

    gettimeofday(&t1, 0);
    Matrix temp(filename);
    gettimeofday(&t2, 0);
    
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to read input:  %3.1f ms \n", time);

    Matrix mat = *(temp.transpose());
    double* x = UniformInit(mat.n);
#ifdef FDEBUG
    mat.print();
#endif
    
    
    gettimeofday(&t1, 0);
    x = RunCPUPowerMethod(&mat, x);
    gettimeofday(&t2, 0);
    
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time for algorithm:  %3.1f ms \n", time);

    ofstream f("output.out");    
    for(int i = 0; i < mat.n; i++)
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

std::string ParseArguments( int argc, char **argv ) {
    if (argc == 2) {
        return string(argv[1]);
    }
    else {
        return string("data.dat");
    }
}


void SerialMatrixMul(double alpha, Matrix *mat, double* x, double* x_new){
    // alpha = 1.0;
    for(int k = 0; k < mat->n; k++)
        x_new[k] = 0;
    for(int i =0; i<mat->n; i++){
        for(int k=mat->col_ind[i]; k<mat->col_ind[i+1]; k++){
            x_new[i] = x_new[i] + alpha*mat->p[k]*x[mat->row_ind[k]];
        }
    }
}