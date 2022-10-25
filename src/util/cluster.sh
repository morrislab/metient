#!/bin/bash
if [ ! $# -eq 2 ]
then
    echo "Usage: $0 <cluster_executable> <machina_sims_data_dir>" >&2
    exit 1
fi

for m in {m5,m8}
do
    if [ ! -e ${2}/${m}_clustered_input ]
    then
	     mkdir ${2}/${m}_clustered_input
    fi

    for p in {mS,S,M,R}
    do
        for f in ${2}/${m}/$p/reads_seed*.tsv
        do
    	    s=$(basename $f .tsv | sed -e s/reads_seed//g)
    	    echo Solving seed $s, pattern $p, anatomical sites $m...
    	    $1 -a 0.001 -b 0.05 ${2}/${m}/$p/reads_seed${s}.tsv > ${2}/${m}_clustered_input/cluster_${p}_seed${s}.tsv 2> ${m}_clustered_input/cluster_${p}_seed${s}.txt
        done
    done
done
