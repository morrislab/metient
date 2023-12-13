#!/bin/bash
if [ ! $# -eq 5 ]
then
    echo "Usage: $0 <ssm_data_dir> <params_data_dir> <orchard_executable> <output_dir> <patient_name>"
    exit 1
fi

source activate orchard
echo Inferring tree for $4...
python $3 --force-monoprimary ${1}/${5}.ssm ${2}/${5}_pyclone_clustered.params.json ${4}/${5}_pyclone_clustered.results.npz
