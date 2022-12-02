#!/bin/bash
# run on turing.cs.princeton.edu

if [ ! $# -eq 1 ]
then
    echo "Usage: $0 <simulate_executable>" >&2
    exit 1
fi

#for mut in {0.1,1,10}
for mut in {1,10}
do
    if [ ! -e $mut ]
    then
        mkdir $mut
    fi
    #for s in {12,23,35,54,89,101,102,105,116,139}
    #for s in {89,101,102,105,116,139}
    for s in {196,198}
    do
        $1 -kP 2 -k 1 -K 5e4 -D 2e-7 -mut ${mut} -m 7 -s $s -P 0.8 -E 0.001 -C 200 -o $mut/ -c ../../coloring.txt -p 1
    done
done

