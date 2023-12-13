#!/bin/bash
if [ ! $# -eq 4 ]
then
    echo "Usage: $0 <ssm_data_dir> <orchard_executable> <output_dir> <patient_name>"
    exit 1
fi

source activate orchard
echo Inferring tree for $4...
python $2 --force-monoprimary ${1}/${4}.ssm ${1}/${4}_clustered.params.json ${3}/${4}_clustered.results.npz
