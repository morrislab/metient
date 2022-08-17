###
#   Creates confidence intervals for all the simulated data from machina
###

#!/bin/bash
if [ ! $# -eq 1 ]
then
    echo "Usage: $0 <sim_data_directory>" >&2
    exit 1
fi


Ms='m5 m8'
Ps='M mS R S'

for m in $Ms ;
do
    for p in $Ps ;
    do
        echo "-----------"
        echo "Processing ${m} with pattern ${p}"

        for read_file in `ls $1/${m}/${p}/reads*`;
        do
            seed_num=$(basename $read_file | tr -d -c 0-9)
            echo $read_file
            cluster_file="$1/clustered_input_${m}/cluster_${p}_seed${seed_num}.txt"
            python create_conf_intervals_from_reads.py $read_file $cluster_file "$1/clustered_input_${m}/" "cluster_${p}_seed${seed_num}.tsv"
        done
    done
done
