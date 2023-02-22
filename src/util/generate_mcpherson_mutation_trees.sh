#!/bin/bash
if [ ! $# -eq 2 ]
then
    echo "Usage: $0 <mcpherson_data_dir> <generatemutationtrees_executable>" >&2
    exit 1
fi

for p in {mS,S,M,R}
do
    echo Generating mutation tree for patient $...
    $2 ${1}/patient${p}_clustered_0.95${s}.tsv > ${1}/patient${p}_mut_trees.txt
    
done
