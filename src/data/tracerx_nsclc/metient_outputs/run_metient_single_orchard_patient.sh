#!/bin/bash

if [ ! $# -eq 4 ]
then
    echo "Usage: $0 <patient> <orchard_tsvs_dir> <orchard_trees_dir> <orchard_output_dir>"
    exit 1
fi


#### Running metient ####

source activate met

echo Running metient for $1...

python run_metient_orchard_patient.py ${1} ${2} ${3} ${4}
