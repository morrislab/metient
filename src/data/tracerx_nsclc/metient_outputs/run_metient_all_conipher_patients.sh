#!/bin/bash
if [ ! $# -eq 3 ]
then
    echo "Usage: $0 <conipher_tsvs_dir> <conipher_trees_dir> <conipher_output_dir>"
    exit 1
fi

for f in ${1}/*.tsv
do
    p=$(basename $f .tsv | sed -e s/_clusterd_SNVS.tsv//g)
    echo "Submitting metient_conipher_job_$p"
    #bsub -J "metient_conipher_job_$p" -n 8 -W 20:00 -o output%I.log -e error_%I.log ./run_metient_single_conipher_patient.sh ${p} ${1} ${2} ${3}
done
