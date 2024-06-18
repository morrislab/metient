#!/bin/bash

if [ ! $# -eq 13 ]
then
    echo "Usage: $0 <machina_sims_data_dir> <num_sites> <mig_type> <seed> <mig_weight> <comig_weight> <seed_weight> <gen_dist_weight> <weight_init_primary> <batch_size> <mode> <solve_polys> <out_dir>"
    exit 1
fi


#### Running metient ####

source activate met

echo "python predict_single_simulated_tree.py ${1} --num_sites ${2} --mig_type ${3} --seed ${4} --wmig ${5} --wcomig ${6} --wseed ${7} --wgen ${8}  --wip ${9} --bs ${10} --mode ${11} --solve_polys ${12} --out_dir ${13}"

python predict_single_simulated_tree.py ${1} --num_sites ${2} --mig_type ${3} --seed ${4} --wmig ${5} --wcomig ${6} --wseed ${7} --wgen ${8}  --wip ${9} --bs ${10} --mode ${11} --solve_polys ${12} --out_dir ${13}
