
[McPherson et. al.](https://www.nature.com/articles/ng.3573#Fig1)

** See `src/jupyter_notebooks/preprocess_mcpherson_ovarian.ipynb` for how these files were generated

* `supplement_table_2.csv`, `supplement_table_7.csv`, `supplement_table_10.csv` are taken directly from [McPherson et. al.](https://www.nature.com/articles/ng.3573#Sec18)
* `pyclone_preprocessing/` contains csvs produced for input into [PyClone](https://github.com/Roth-Lab/pyclone) for clustering, as well as the output from the method. 
	* `pyclone_preprocessing/patient*_pyclone_clusters.txt` contains the PyClone clusters for that patient. Each line number indicates the PyClone cluster ID, with all the mutations in that cluster on that line.
* `patient*_clustered_0.95.tsv` contains mutation data pooled by clusters, creating conf intervals and summed ref/var for each cluster (`src/utils/create_conf_intervals_from_reads`)
