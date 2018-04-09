#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <float.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include "serial_types.h"
#include "find_top_k.h"

using namespace std;

const float EPS = 0.000001;
double alpha = 0.8;

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
    double x_norm;
    x_new = new double[P->n];
    cout << "Checkpoint" << endl;
	while(abs(lambda - oldLambda) > EPS)
	{
		oldLambda = lambda;
        SerialMatrixMul(alpha, P, x, x_new);
        double x_norm = 0;
        for(int i =0; i<P->n; i++){
            x_norm += abs(x_new[i]);
        }
        cout<<x_norm<<endl;
        for(int i =0; i<P->n; i++){
            x_new[i] = x_new[i]/x_norm;
        }
        // PrintArray(x_new,P->n);
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

double* RandomInit(int n) {
    double *x = new double[n];
    srand(0);
    for (int i = 0; i < n; i++) {
        x[i] = (rand() % 100) / 100.0; 
    }
    return x;
}

int main(int argc, char** argv)
{
    std::string filename;
    filename = ParseArguments(argc, argv);

    Matrix mat(filename);

    double* x = RandomInit(mat.n);
    double x_norm = 0;
    for(int i =0; i<mat.n; i++){
        x_norm += abs(x[i]);
    }
    for(int i =0; i<mat.n; i++){
        x[i] = x[i]/x_norm;
    }
#ifdef FDEBUG
    mat.print();
#endif
    
    x = RunCPUPowerMethod(&mat, x);

    ofstream f("output.out");    
    for(int i = 0; i < mat.n; i++)
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
    PrintArray(x,mat->n);
    PrintArray(x_new,mat->n);
}