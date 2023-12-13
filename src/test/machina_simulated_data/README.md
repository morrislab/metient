python predict_all_simulated_trees.py ../../data/machina_sims/ bs_256_gd_r1_wip_10032023 --bs 256 --gen 1.0 --wip; 
to run machina:

export GRB_LICENSE_KEY="/home/koyyald/gurobi.lic"
export GRB_LICENSE_FILE="/home/koyyald/gurobi.lic"
export LD_LIBRARY_PATH=/lila/home/koyyald/mambaforge/envs/machina/lib/libstdc++.so.6:$LD_LIBRARY_PATH
export LD_PRELOAD=/lila/home/koyyald/mambaforge/envs/machina/lib/libstdc++.so.6
./machina_pmh_ti_timing.sh /data/morrisq/divyak/machina-linux-binaries/pmh_ti ../../../../machina/data/sims/ ../../data/machina_sims/
