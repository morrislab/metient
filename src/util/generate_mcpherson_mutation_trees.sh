#!/bin/bash
if [ ! $# -eq 2 ]
then
    echo "Usage: $0 <mcpherson_data_dir> <generatemutationtrees_executable>" >&2
    exit 1
fi

for p in {1,2,3,4,7,9,10}
do
    echo Generating mutation tree for patient ${p}...
    $2 ${1}/patient${p}_clustered_0.95.tsv > ${1}/patient${p}_mut_trees.txt
    
done
