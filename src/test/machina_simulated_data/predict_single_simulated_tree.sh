#!/bin/bash

if [ ! $# -eq 15 ]
then
    echo "Usage: $0 <machina_sims_data_dir> <num_sites> <mig_type> <seed> <tree_num> <data_fit_weight> <mig_weight> <comig_weight> <seed_weight> <reg_weight> <gen_dist_weight> <weight_init_primary> <batch_size> <lr_sched> <out_dir>"
    exit 1
fi


#### Running metient ####

source activate met

python predict_single_simulated_tree.py ${1} --num_sites ${2} --mig_type ${3} --seed ${4} --tree_num ${5} --wdata_fit ${6} --wmig ${7} --wcomig ${8} --wseed ${9} --wreg ${10} --wgen ${11}  --wip ${12} --bs ${13} --lr_sched ${14} --out_dir ${15}
