#!/bin/bash
if [ ! $# -eq 3 ]
then
    echo "Usage: $0 <ssm_data_dir> <pairtree_clustering_executable> <patient_name>"
    exit 1
fi
source activate pairtree
echo Clustering mutations for $3...
$2 ${1}/${3}.ssm ${1}/${3}.params.json ${1}/${3}_clustered.params.json --model linfreq
