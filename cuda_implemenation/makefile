LDFLAGS := -lcusparse -lcublas
CCFLAGS := -g -G

out: main.o Utilities.o read.o types.o cuda_subtract.o cuda_norm.o cuda_matmul.o cuda_divide.o
	nvcc -o out main.o Utilities.o read.o types.o cuda_matmul.o cuda_subtract.o cuda_norm.o cuda_divide.o $(LDFLAGS) $(CCFLAGS)

main.o: main.cu
	nvcc main.cu -c main.o $(LDFLAGS) $(CCFLAGS)

Utilities.o: Utilities.cu
	nvcc Utilities.cu -c Utilities.o $(LDFLAGS) $(CCFLAGS)

read.o: read.cu
	nvcc read.cu -c read.o $(LDFLAGS) $(CCFLAGS)

types.o: types.cu types.cuh
	nvcc types.cu -c types.o $(LDFLAGS) $(CCFLAGS)

cuda_matmul.o: cuda_matmul.cu types.o
	nvcc cuda_matmul.cu -c cuda_matmul.o $(LDFLAGS) $(CCFLAGS)

cuda_norm.o : cuda_norm.cu
	nvcc cuda_norm.cu -c cuda_norm.o $(LDFLAGS) $(CCFLAGS)

cuda_subtract.o : cuda_subtract.cu
	nvcc cuda_subtract.cu -c cuda_subtract.o $(LDFLAGS) $(CCFLAGS)

cuda_divide.o: cuda_divide.cu
	nvcc cuda_divide.cu -c cuda_divide.o $(LDFLAGS) $(CCFLAGS)

clean:
	rm -f debug pagerank *.o out
