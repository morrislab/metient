#!/bin/bash
now="$(date)"
printf "Start date and time %s\n" "$now"

if [ ! $# -eq 1 ]
then
    echo "Usage: $0 <pmh_ti_executable>" >&2
    exit 1
fi
for m in {m5,m8}
do
    if [ ! -e input_${m} ]
    then
	mkdir input_${m}
    fi

    if [ ! -e mut_trees_${m} ]
    then
	mkdir mut_trees_${m}
    fi

    if [ ! -e output_${m} ]
    then
	mkdir output_${m}
    fi
    
    for p in {mS,S,M,R}
    do
        for f in ../../../data/sims/${m}/$p/reads_seed*.tsv
        do
    	    s=$(basename $f .tsv | sed -e s/reads_seed//g)

	    echo Solving seed $s, pattern $p, anatomical sites $m...
	    
	    if [ ! -e output_${m}/${p}_${s} ]
	    then
		mkdir output_${m}/${p}_${s}
	    fi
	    seed_start=`date +%s.%N`
            $1 -p P -c ../../../data/sims/coloring.txt -m 1,2,3 -t 8 -o output_${m}/${p}_${s}/ -F input_${m}/cluster_${p}_seed${s}.tsv -barT mut_trees_${m}/mut_trees_${p}_seed${s}.txt > output_${m}/${p}_${s}.txt
            seed_end=`date +%s.%N`
	    seed_time=$( echo "$seed_end - $seed_start" | bc -l )
            echo "Runtime for seed ${s}, pattern ${p}, anatomical sites ${m}: ${seed_time}"
	done
    done
done

now="$(date)"
printf "end date and time %s\n" "$now"
