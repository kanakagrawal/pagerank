out: cusparseDcsrmv_example.o Utilities.o read.o
	nvcc -o out cusparseDcsrmv_example.o Utilities.o read.o -lcublas -lcusparse

cusparseDcsrmv_example.o: cusparseDcsrmv_example.cu
	nvcc cusparseDcsrmv_example.cu -c cusparseDcsrmv_example.o -lcusparse -lcublas

Utilities.o: Utilities.cu
	nvcc Utilities.cu -c Utilities.o -lcusparse -lcublas

read.o: read.cu
	nvcc read.cu -c read.o

pagerank: main.cu
	nvcc main.cu -o pagerank

debug: main.cu
	nvcc -DDEBUG main.cu -o debug

clean:
	rm -f debug pagerank *.o out