#!/bin/bash
if [ ! $# -eq 2 ]
then
    echo "Usage: $0 <machina_sims_data_dir> <generatemutationtrees_executable>" >&2
    exit 1
fi

for m in {m5,m8}
do

    if [ ! -e ${1}/mut_trees_${m} ]
    then
	     mkdir ${1}/mut_trees_${m}
    fi

    for p in {mS,S,M,R}
    do
        for f in ${1}/${m}/$p/reads_seed*.tsv
        do
    	      s=$(basename $f .tsv | sed -e s/reads_seed//g)

      	    echo Solving seed $s, pattern $p, anatomical sites $m...
      	    $2 ${1}/${m}_clustered_input/cluster_${p}_seed${s}.tsv > ${1}/mut_trees_${m}/mut_trees_${p}_seed${s}.txt
        done
    done
done
