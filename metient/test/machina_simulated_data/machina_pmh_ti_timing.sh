#!/bin/bash
now="$(date)"
printf "Start date and time %s\n" "$now"

if [ ! $# -eq 5 ]
then
    echo "Usage: $0 <pmh_ti_executable> <original_repo_machina_sims_dir> <metient_machina_sims_dir> <poly_res> <output_dir>" >&2
    exit 1
fi

poly_res_input=$4

# Convert user input to lowercase for case-insensitive comparison
use_poly_res=$(echo "$poly_res_input" | tr '[:upper:]' '[:lower:]')

if [ "$use_poly_res" == "true" ]; then
    echo "Using polytomy resolution."
elif [ "$use_poly_res" == "false" ]; then
    echo "Not using polytomy resolution."
else
    echo "Invalid input. Please enter true or false."
    exit 1
fi

if [ ! -e ${5} ]
    then
    mkdir ${5}
fi

for m in {m5,m8}
do

    if [ ! -e ${5}/output_${m} ]
    then
	mkdir ${5}/output_${m}
    fi
    
    for p in {mS,S,M,R}
    do
        for f in ${2}/${m}/$p/reads_seed*.tsv
        do
    	    s=$(basename $f .tsv | sed -e s/reads_seed//g)
    	    
    	    if [ ! -e ${5}/output_${m}/${p}_${s} ]
    	    then
    		mkdir ${5}/output_${m}/${p}_${s}
    	    fi

            num_trees=$(find "${3}/${m}_split_mut_trees/" -type f -name "mut_trees_${p}_seed${s}_tree*.txt" | wc -l)

            for ((i=0;i<$num_trees;i++)); do
                echo Solving seed $s, pattern $p, anatomical sites $m, tree $i...

                seed_start=`date +%s.%N`

                if [ "$use_poly_res" == "true" ]; then
                    $1 -p P -c ${2}/coloring.txt -t 1 -o ${5}/output_${m}/${p}_${s}/ -F ${3}/${m}_clustered_input/cluster_${p}_seed${s}.tsv -barT ${3}/${m}_split_mut_trees/mut_trees_${p}_seed${s}_tree${i}.txt -log > ${5}/output_${m}/${p}_${s}.txt
                elif [ "$use_poly_res" == "false" ]; then
                    $1 -noPR -p P -c ${2}/coloring.txt -t 1 -o ${5}/output_${m}/${p}_${s}/ -F ${3}/${m}_clustered_input/cluster_${p}_seed${s}.tsv -barT ${3}/${m}_split_mut_trees/mut_trees_${p}_seed${s}_tree${i}.txt -log > ${5}/output_${m}/${p}_${s}.txt
                fi
                seed_end=`date +%s.%N`
                seed_time=$( echo "$seed_end - $seed_start" | bc -l )
                echo "Runtime for seed ${s}, pattern ${p}, anatomical sites ${m}: ${seed_time}"
            done
	    
	    done
    done
done

now="$(date)"
printf "end date and time %s\n" "$now"
