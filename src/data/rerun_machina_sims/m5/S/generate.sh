#!/bin/bash

if [ ! $# -eq 1 ]
then
    echo "Usage: $0 <simulate_executable>" >&2
    exit 1
fi

#$1 -kP 2 -k 1 -D 2e-7 -m 4 -p 0 -s 0 -N 10 -C 200 -o . -c ../../coloring.txt

for s in {17,23,25,31,32,35,40,49,62,81}
do
    $1 -kP 2 -k 1 -D 2e-7 -m 4 -p 0 -s $s -C 200 -E 0.001 -P 1 -o . -c ../../../sims/coloring.txt
done


