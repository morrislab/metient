`bsub -n 8 -W 20:00 -R rusage[mem=8] -o output_calibrate_12052023.log -e error_calibrate_12052023.log python calibrate_conipher_patients.py ../patient_data/pyclone_clustered/ ../conipher_outputs/TreeBuilding/ pyclone_conipher_calibrate_12052023`
`./run_metient_all_conipher_patients.sh ../patient_data/pyclone_clustered/ ../conipher_outputs/TreeBuilding/ ./pyclone_clustered_conipher_trees/`
`./run_metient_all_orchard_patients.sh ../patient_data/pyclone_clustered/ ../orchard_trees/pyclone_clustered/ ./pyclone_clustered_orchard_trees/`

