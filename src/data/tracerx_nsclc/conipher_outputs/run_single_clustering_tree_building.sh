#!/bin/bash

if [ ! $# -eq 4 ]
then
    echo "Usage: $0 <input_dir> <conipher_script_dir> <output_dir> <patient_name>"
    exit 1
fi

################################################################################## Input parameters
###################################################################################################

case_id=${4}
scriptDir=${2}/src/
inputTSV=${1}/${4}.tsv
outDir=${3}


############################################################### Running clustering and treebuilding
###################################################################################################

source activate conipher


echo Running clustering and tree building for $4...

clusteringDir=${outDir}"/Clustering/"
treeDir=${outDir}"/TreeBuilding/"

mkdir -p ${clusteringDir}
mkdir -p ${treeDir}

Rscript ${scriptDir}run_clustering.R \
--case_id ${case_id} \
--script_dir ${scriptDir} \
--input_tsv ${inputTSV} \
--working_dir ${clusteringDir} \
--nProcs 1

Rscript ${scriptDir}run_treebuilding.R \
--input_tsv ${clusteringDir}${case_id}".SCoutput.CLEAN.tsv" \
--out_dir ${treeDir} \
--script_dir ${scriptDir} \
--prefix CRUK

# conda deactivate
