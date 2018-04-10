#!/bin/bash

for f in ../mpj/PageRankMPI/Inputfile/*; do
	b=$(basename $f)
 	echo "File -> $b"
 	python3 mpi_to_cuda_data.py $f
 	mv cuda_$b cuda_data/
done