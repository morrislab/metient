#!/bin/bash
if [ ! $# -eq 3 ]
then
    echo "Usage: $0 <ssm_data_dir> <output_dir> <orchard_executable>"
    exit 1
fi

for f in ${1}/*.ssm
do
    p=$(basename $f .ssm | sed -e s/ssm//g)
    echo Inferring tree for $p...
    python $3 --force-monoprimary ${1}/${p}.ssm ${1}/${p}_clustered.params.json ${2}/${p}_clustered_muts.results.npz

done
