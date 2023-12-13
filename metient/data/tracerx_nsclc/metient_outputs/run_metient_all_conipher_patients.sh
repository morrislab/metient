#!/bin/bash
if [ ! $# -eq 3 ]
then
    echo "Usage: $0 <conipher_tsvs_dir> <conipher_trees_dir> <conipher_output_dir>"
    exit 1
fi

rm -rf ${3}; 
mkdir -p ${3}"/max_pars/"
mkdir -p ${3}"/max_pars_genetic_distance/"

for f in ${1}/*.tsv
do
    p=$(basename $f .tsv | sed -e "s/_clustered_SNVs//")
    echo "Submitting metient_conipher_job_$p"
    bsub -J "metient_conipher_job_$p" -n 8 -W 20:00 -o output_metient_conipher.log -e error_metient_conipher.log ./run_metient_single_conipher_patient.sh ${p} ${1} ${2} ${3}
done
