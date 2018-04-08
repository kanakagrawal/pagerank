#include <stdio.h>
#include <math.h>
#include "device_functions.h"

const int BLOCK_SIZE =32;


// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* h_NormW = NULL;
float* d_NormW = NULL;

// Variables to change
int GlobalSize = 50000;
int BlockSize = 32;
const float EPS = 0.000001;

// create and start timer as unsigned integer
unsigned int timer_mem = 0;
unsigned int timer_total = 0;
unsigned int timer_GPU = 0;
unsigned int timer_CPU=0;

unsigned int timer_Av = 0;
unsigned int timer_Norm = 0;
unsigned int timer_Lamda=0;


// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
void PrintArray(float*, int);
float CPUReduce(float*, int);
void ParseArguments(int, char**);

void CPU_AvProduct()
{
	int N = GlobalSize;
	int matIndex =0;
    for(int i=0;i<N;i++)
	{
		h_VecW[i] = 0;
		for(int j=0;j<N;j++)
		{
			matIndex = i*N + j;
			h_VecW[i] += h_MatA[matIndex] * h_VecV[j];
			
		}
	}
}


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

float CPU_ComputeLamda()
{
	int N = GlobalSize;
	float lamda =0;
	for(int i=0;i<N;i++)
		lamda += h_VecV[i] * h_VecW[i];
	
	return lamda;
}



void RunCPUPowerMethod()
{
	printf("*************************************\n");
	float oldLamda =0;
	float lamda=0;
	
	//AvProduct
	CPU_AvProduct();
	
	//power loop
	for (int i=0;i<100;i++)
	{
		CPU_NormalizeW();
		CPU_AvProduct();
		lamda= CPU_ComputeLamda();
		printf("CPU lamda at %d: %f \n", i, lamda);
		// If residual is lass than epsilon break
		if(abs(oldLamda - lamda) < EPS)
			break;
		oldLamda = lamda;	
	
	}
	printf("*************************************\n");
	
}


int main(int argc, char** argv)
{
    ParseArguments(argc, argv);
		
    int N = GlobalSize;
    printf("Matrix size %d X %d \n", N, N);
    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);
    size_t norm_size = sizeof(float);
    //float CPU_result = 0.0, GPU_result = 0.0;

    // Allocate input matrix in host memory
    h_MatA = (float*)malloc(mat_size);
    if (h_MatA == 0) 
      Cleanup();

    // Allocate initial vector V in host memory
    h_VecV = (float*)malloc(vec_size);
    if (h_VecV == 0) 
      Cleanup();

    // Allocate W vector for computations
    h_VecW = (float*)malloc(vec_size);
    if (h_VecW == 0) 
      Cleanup();

   h_NormW = (float*)malloc(norm_size);

    // Initialize input matrix
    UploadArray(h_MatA, N);
    InitOne(h_VecV,N);
	
    RunCPUPowerMethod();
}

void Cleanup(void)
{
    // Free device memory
    if (d_MatA)
        cudaFree(d_MatA);
    if (d_VecV)
        cudaFree(d_VecV);
    if (d_VecW)
        cudaFree(d_VecW);
	if (d_NormW)
		cudaFree(d_NormW);
		
    // Free host memory
    if (h_MatA)
        free(h_MatA);
    if (h_VecV)
        free(h_VecV);
    if (h_VecW)
        free(h_VecW);
     if (h_NormW)
        free(h_NormW);
		

	
    cudaThreadExit();
    
    exit(0);
}

// Allocates an array with zero value.
void InitOne(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
	data[0]=1;
}

void UploadArray(float* data, int n)
{
   int total = n*n;
   int value=1;
    for (int i = 0; i < total; i++){
    	data[i] = (int) (rand() % (int)(101));//1;//value;
	value ++; if(value>n) value =1;
    }
}
void PrintArray(float* data, int n)
{
    for (int i = 0; i < n; i++)
        printf("[%d] => %f\n",i,data[i]);
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  GlobalSize = atoi(argv[i+1]);
		  i = i + 1;
        }
        if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0) {
                  BlockSize = atoi(argv[i+1]);
		  i = i + 1;
	}
    }
}