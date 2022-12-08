#!/bin/bash

#BSUB -J max_pars_only_12082022
#BSUB -W 10:00
#BSUB -n 8
#BSUB -R rusage[mem=2]
#BSUB -e %J.err
#BSUB -o %J.out

conda activate met
python -u predict_simulated_data_vertex_labeling.py ../../data/machina_sims/ max_pars_only_12082022 --data_fit=1.0 --mig=10.0 --comig=5.0 --reg=1.0 --gen=0.0 --cores=8 > predict_max_pars_only_12082022.txt