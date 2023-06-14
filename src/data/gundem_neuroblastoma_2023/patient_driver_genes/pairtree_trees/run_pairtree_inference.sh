#!/bin/bash
if [ ! $# -eq 4 ]
then
    echo "Usage: $0 <ssm_data_dir> <output_dir> <pairtree_executable> <plottree_executable>"
    exit 1
fi

for f in ${1}/*.ssm
do
    p=$(basename $f .ssm | sed -e s/ssm//g)
    echo Inferring tree for $p...
    $3 --params ${1}/${p}.params.json ${1}/${p}.ssm ${2}/${p}.results.npz
    $4 --runid ${p} ${1}/${p}.ssm ${1}/${p}.params.json ${2}/${p}.results.npz ${2}/${p}.results.html

done
