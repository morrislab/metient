#!/bin/bash

if [ ! $# -eq 4 ]
then
    echo "Usage: $0 <patient> <conipher_tsvs_dir> <conipher_trees_dir> <conipher_output_dir>"
    exit 1
fi


#### Running metient ####

source activate met

echo Running metient for $1...

python run_metient_conipher_patient.py ${1} ${2} ${3} ${4}
