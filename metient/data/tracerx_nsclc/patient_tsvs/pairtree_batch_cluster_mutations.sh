#!/bin/bash
#BSUB -J "batch_job[1-58]"    # Adjust the job array range based on the number of files
#BSUB -n 2
#BSUB -q cpu_queue
#BSUB -o batch_output_%I.log
#BSUB -e batch_error_%I.log
#BSUB -W 6:00
#BSUB -R rusage[mem=8]

if [ ! $# -eq 2 ]
then
    echo "Usage: $0 <ssm_data_dir> <pairtree_clustering_executable>"
    exit 1
fi

input_files=('CRUK0485',
 'CRUK0487',
 'CRUK0495',
 'CRUK0496',
 'CRUK0497',
 'CRUK0510',
 'CRUK0514',
 'CRUK0516',
 'CRUK0519',
 'CRUK0524',
 'CRUK0528',
 'CRUK0530',
 'CRUK0537',
 'CRUK0543',
 'CRUK0552',
 'CRUK0557',
 'CRUK0559',
 'CRUK0567',
 'CRUK0572',
 'CRUK0584',
 'CRUK0587',
 'CRUK0589',
 'CRUK0590',
 'CRUK0596',
 'CRUK0598',
 'CRUK0609',
 'CRUK0617',
 'CRUK0620',
 'CRUK0625',
 'CRUK0636',
 'CRUK0640',
 'CRUK0666',
 'CRUK0667',
 'CRUK0691',
 'CRUK0693',
 'CRUK0698',
 'CRUK0702',
 'CRUK0707',
 'CRUK0714',
 'CRUK0718',
 'CRUK0719',
 'CRUK0721',
 'CRUK0722',
 'CRUK0730',
 'CRUK0733',
 'CRUK0736',
 'CRUK0737',
 'CRUK0742',
 'CRUK0745',
 'CRUK0748',
 'CRUK0762',
 'CRUK0766',
 'CRUK0769',
 'CRUK0794',
 'CRUK0799',
 'CRUK0810',
 'CRUK0817',
 'CRUK0872')

conda activate pairtree

for ((i=($LSB_JOBINDEX - 1); i<${#input_files[@]}; i+=1))
do
    p="${input_files[$i]}"
    echo Clustering mutations for $p...
    $2 ${1}/${p}.ssm ${1}/${p}.params.json ${1}/${p}_clustered.params.json --model linfreq
done
