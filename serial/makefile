CC := g++

serial: main.o read.o types.o find_top_k.o
	$(CC) -o serial main.o read.o types.o find_top_k.o

main.o: serial.cpp
	$(CC) serial.cpp -c -o main.o

read.o: read.cpp
	$(CC) -std=c++11 read.cpp -c

types.o: types.cpp types.h
	$(CC) types.cpp -c 

find_top_k.o: find_top_k.cpp find_top_k.h
	$(CC) find_top_k.cpp -c

clean:
	rm -f serial *.o a.out