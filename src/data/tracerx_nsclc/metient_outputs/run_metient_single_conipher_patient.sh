#!/bin/bash

if [ ! $# -eq 4 ]
then
    echo "Usage: $0 <patient> <conipher_tsvs_dir> <conipher_trees_dir> <conipher_output_dir>"
    exit 1
fi


#### Running metient ####

source activate met

echo Running metient for $1...

mkdir -p ${4}"/max_pars/"
mkdir -p ${4}"/max_pars_wip/"
mkdir -p ${4}"/max_pars_genetic_distance/"
mkdir -p ${4}"/max_pars_genetic_distance_wip/"

python run_conipher_patient.py ${1} ${2} ${3} ${4}