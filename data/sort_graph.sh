#!/bin/bash

# python generate_data.py $1 $2 $3
tail -n $2 $3 | sort -k2n -k1n > temp
uniq temp > part2
count=$(wc -l part2 | cut -f 1 -d " ")

head -n $(($1+1)) $3 > temp
numnodes=$(head -n 1 temp | cut -f 1 -d " ")
echo "$numnodes $count" > part1.1
tail -n $1 temp > part1.2

cat part1.1 part1.2 part2 > $3

rm -f part1.1 part1.2 part2 temp
