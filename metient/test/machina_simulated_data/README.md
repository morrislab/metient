python predict_all_simulated_trees.py ../../data/machina_sims/ bs_256_gd_r1_wip_10032023 --bs 256 --gen 1.0 --wip; 
to run machina:

export GRB_LICENSE_KEY="/home/koyyald/gurobi.lic"
export GRB_LICENSE_FILE="/home/koyyald/gurobi.lic"
export LD_LIBRARY_PATH=/lila/home/koyyald/mambaforge/envs/machina/lib/libstdc++.so.6:$LD_LIBRARY_PATH
export LD_PRELOAD=/lila/home/koyyald/mambaforge/envs/machina/lib/libstdc++.so.6
./machina_pmh_ti_timing.sh /data/morrisq/divyak/machina-linux-binaries/pmh_ti ../../../../machina/data/sims/ ../../data/machina_sims/


Commands to run on simulated data:
bsub -n 8 -W 40:00 -R 'rusage[mem=8] span[hosts=1]' -o output_bs1024_calibrate_solvepoly_wip_allcombos_02182025.log -e error_bs1024_calibrate_solvepoly_wip_allcombos_02182025.log python predict_all_simulated_trees_calibrate_new_split.py ../../data/machina_sims/ bs1024_calibrate_solvepoly_wip_allcombos_02182025 --gen 1.0 --solve_polys --bs 1024 --wip; 
bsub -n 8 -W 40:00 -R 'rusage[mem=8] span[hosts=1]' -o output_bs1024_calibrate_wip_allcombos_02182025.log -e error_bs1024_calibrate_wip_allcombos_02182025.log python predict_all_simulated_trees_calibrate_new_split.py ../../data/machina_sims/ bs1024_calibrate_wip_allcombos_02182025 --gen 1.0  --bs 1024 --wip; 

python predict_all_simulated_trees_evaluate.py ../../data/machina_sims/ bs1024_evaluate_solvepoly_wip_allcombos_02182025 --gen 0.0 --solve_polys --bs 1024 --wip; python predict_all_simulated_trees_evaluate.py ../../data/machina_sims/ bs1024_evaluate_wip_allcombos_02182025 --gen 0.0 --bs 1024 --wip; 

python predict_all_simulated_trees_evaluate.py ../../data/machina_sims/ bs1024_evaluate_solvepoly_gd_only_wip_allcombos_02182025 --gen 1.0 --mig 0 --comig 0 --seed 0  --solve_polys --bs 1024 --wip; python predict_all_simulated_trees_evaluate.py ../../data/machina_sims/ bs1024_evaluate_gd_only_wip_allcombos_02182025 --gen 1.0 --mig 0 --comig 0 --seed 0 --bs 1024 --wip;