out: main.o Utilities.o read.o types.o cuda_subtract.o cuda_norm.o cuda_matmul.o
	nvcc -o out main.o Utilities.o read.o types.o cuda_matmul.o cuda_subtract.o cuda_norm.o -lcublas -lcusparse

main.o: main.cu
	nvcc main.cu -c main.o -lcublas -lcusparse

cuda_matmul.o: cuda_matmul.cu types.o
	nvcc cuda_matmul.cu -c cuda_matmul.o types.o -lcusparse -lcublas

Utilities.o: Utilities.cu
	nvcc Utilities.cu -c Utilities.o -lcusparse -lcublas

read.o: read.cu
	nvcc read.cu -c read.o

types.o: types.cu types.cuh
	nvcc types.cu -c types.o

cuda_norm.o : cuda_norm.cu
	nvcc cuda_norm.cu -c cuda_norm.o -lcusparse -lcublas

cuda_subtract.o : cuda_subtract.cu
	nvcc cuda_subtract.cu -c cuda_subtract.o -lcusparse -lcublas

pagerank: main.cu Utilities.o
	nvcc main.cu -o pagerank

debug: main.cu
	nvcc -DDEBUG main.cu -o debug

clean:
	rm -f debug pagerank *.o out
