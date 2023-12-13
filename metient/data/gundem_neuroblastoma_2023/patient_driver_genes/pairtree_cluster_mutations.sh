#!/bin/bash
if [ ! $# -eq 2 ]
then
    echo "Usage: $0 <ssm_data_dir> <pairtree_clustering_executable>"
    exit 1
fi

for f in ${1}/*.ssm
do
    p=$(basename $f .ssm | sed -e s/ssm//g)
    echo Clustering mutations for $p...
    $2 ${1}/${p}.ssm ${1}/${p}.params.json ${1}/${p}_clustered.params.json --model pairwise

done
