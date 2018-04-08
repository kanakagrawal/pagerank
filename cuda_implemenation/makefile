out: cuda_matmul.o Utilities.o read.o types.o
	nvcc -o out cuda_matmul.o Utilities.o read.o types.o -lcublas -lcusparse

cuda_matmul.o: cuda_matmul.cu types.o
	nvcc cuda_matmul.cu -c cuda_matmul.o types.o -lcusparse -lcublas

Utilities.o: Utilities.cu
	nvcc Utilities.cu -c Utilities.o -lcusparse -lcublas

read.o: read.cu
	nvcc read.cu -c read.o

types.o: types.cu types.cuh
	nvcc types.cu -c types.o

test_norm : 
	nvcc cuda_norm.cu Utilities.cu -lcusparse -lcublas

test_sub : 
	nvcc cuda_subtract.cu Utilities.cu -lcusparse -lcublas

pagerank: main.cu Utilities.o
	nvcc main.cu -o pagerank

debug: main.cu
	nvcc -DDEBUG main.cu -o debug

clean:
	rm -f debug pagerank *.o out
