out: cusparseDcsrmv_example.o Utilities.o read.o
	nvcc -o out cusparseDcsrmv_example.o Utilities.o read.o -lcublas -lcusparse

cusparseDcsrmv_example.o:
	nvcc cusparseDcsrmv_example.cu -c cusparseDcsrmv_example.o -lcusparse -lcublas

Utilities.o:
	nvcc Utilities.cu -c Utilities.o -lcusparse -lcublas

read.o:
	nvcc read.cu -c read.o

pagerank: main.cu
	nvcc main.cu -o pagerank

debug: main.cu
	nvcc -DDEBUG main.cu -o debug

clean:
	rm -f debug pagerank *.o