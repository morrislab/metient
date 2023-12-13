#!/bin/bash

if [ ! $# -eq 2 ]
then
    echo "Usage: $0 <ssm_data_dir> <pairtree_clustering_executable>"
    exit 1
fi

for f in ${1}/*.ssm
do
    p=$(basename $f .ssm | sed -e s/ssm//g)
    if [ -f "${1}/${p}_clustered.params.json" ]; then
        echo "Skipping $p, already clustered."
    else
        echo "Submitting batch_job_$p"
        bsub -J "batch_job_$p" -n 2 -W 5:00 -o output_%I.log -e error_%I.log ./run_single_clustering.sh ${1} ${2} ${p}
    fi
done
