#!/bin/bash
if [ ! $# -eq 3 ]
then
    echo "Usage: $0 <orchard_tsvs_dir> <orchard_trees_dir> <orchard_output_dir>"
    exit 1
fi

for f in ${2}/*.results.npz
do
    p=$(basename $f .tsv | sed -e "s/.results.npz//")
    echo "Submitting metient_orchard_job_$p"
    #bsub -J "metient_orchard_job_$p" -n 8 -W 20:00 -o output_metient_orchard.log -e error_metient_orchard.log ./run_metient_single_orchard_patient.sh ${p} ${1} ${2} ${3}
done
