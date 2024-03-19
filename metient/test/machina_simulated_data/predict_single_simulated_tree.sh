#!/bin/bash

if [ ! $# -eq 15 ]
then
    echo "Usage: $0 <machina_sims_data_dir> <num_sites> <mig_type> <seed> <data_fit_weight> <mig_weight> <comig_weight> <seed_weight> <reg_weight> <gen_dist_weight> <weight_init_primary> <batch_size> <mode> <solve_polys> <out_dir>"
    exit 1
fi


#### Running metient ####

source activate met

echo "python predict_single_simulated_tree.py ${1} --num_sites ${2} --mig_type ${3} --seed ${4} --wdata_fit ${5} --wmig ${6} --wcomig ${7} --wseed ${8} --wreg ${9} --wgen ${10}  --wip ${11} --bs ${12} --mode ${13} --solve_polys ${14} --out_dir ${15}"

python predict_single_simulated_tree.py ${1} --num_sites ${2} --mig_type ${3} --seed ${4} --wdata_fit ${5} --wmig ${6} --wcomig ${7} --wseed ${8} --wreg ${9} --wgen ${10}  --wip ${11} --bs ${12} --mode ${13} --solve_polys ${14} --out_dir ${15}
