#!/bin/bash

if [ ! $# -eq 4 ]
then
    echo "Usage: $0 <patient> <orchard_tsvs_dir> <orchard_trees_dir> <orchard_output_dir>"
    exit 1
fi


#### Running metient ####

source activate met

echo Running metient for $1...

rm -rf  ${4}"/max_pars/"; mkdir -p ${4}"/max_pars/"
rm -rf ${4}"/max_pars_wip/"; mkdir -p ${4}"/max_pars_wip/"
rm -rf ${4}"/max_pars_genetic_distance/"; mkdir -p ${4}"/max_pars_genetic_distance/"
rm -rf ${4}"/max_pars_genetic_distance_wip/"; mkdir -p ${4}"/max_pars_genetic_distance_wip/"

python run_metient_orchard_patient.py ${1} ${2} ${3} ${4}
