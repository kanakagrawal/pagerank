#!/bin/bash

cd cuda_implemenation/
make clean
make
cd ..

cd serial/
make clean
make
cd ..

for f in data/cuda_data/*; do
	b=$(basename $f)
 	echo "File -> $b"
 	./cuda_implemenation/out $f > timings/cuda_$b
 	./serial/serial_out $f > timings/serial_$b
done