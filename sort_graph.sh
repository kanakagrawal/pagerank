#!/bin/bash

tail -n 600000 data | sort -n -t ' ' -k 1 | sort -n -t ' ' -k 2  > data_part2
head -n 100001 data > data_part1
cat data_part1 data_part2 > data

# rm -f data_part1 data_part2