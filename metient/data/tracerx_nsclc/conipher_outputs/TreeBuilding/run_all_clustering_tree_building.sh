#!/bin/bash
if [ ! $# -eq 3 ]
then
    echo "Usage: $0 <conipher_inputs_dir> <conipher_script_dir> <output_dir> "
    exit 1
fi

for f in ${1}/*.tsv
do
    p=$(basename $f .tsv | sed -e s/tsv//g)
    echo "Submitting conipher_job_$p"
    bsub -J "conipher_job_$p" -n 8 -W 20:00 -o output%I.log -e error_%I.log ./run_single_clustering_tree_building.sh ${1} ${2} ${3} ${p}
done
