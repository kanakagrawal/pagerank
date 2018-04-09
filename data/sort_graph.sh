#!/bin/bash

# python generate_data.py $1 $2 $3
tail -n $2 $3 | sort -k1n -k2n > temp
uniq temp > part2
count=$(wc -l part2 | cut -f 1 -d " ")

head -n $(($1+1)) $3 > temp
x=$(head -n 1 temp | cut -f 1 -d " ")
y=$(head -n 1 temp | cut -f3-4 -d " ")
echo "$x $count $y" > part1.1
tail -n $1 temp > part1.2

cat part1.1 part1.2 part2 > $3

rm -f part1.1 part1.2 part2 temp