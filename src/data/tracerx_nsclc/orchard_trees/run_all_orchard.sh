#!/bin/bash
if [ ! $# -eq 4 ]
then
    echo "Usage: $0 <ssm_data_dir> <params_data_dir> <orchard_executable> <output_dir>"
    exit 1
fi

for f in ${1}/*.ssm
do
    p=$(basename $f .ssm | sed -e s/ssm//g)
    if [ -f "${3}/${p}_clustered.muts.results.npz" ]; then
        echo "Skipping $p, already clustered."
    else
        echo "Submitting orchard_job_$p"
        bsub -J "orchard_job_$p" -n 2 -W 40:00 -R rusage[mem=8] -o output%I.log -e error_%I.log ./run_single_orchard.sh ${1} ${2} ${3} ${4} ${p}
    fi
done
