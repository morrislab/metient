# Metient
<!-- <p align="center">
<img src="metient/logo.png" width="150">
</p> -->

<p align="center">
  <img src="metient/method_overview.png?cachebust=12345" width="1000" style="padding:20px;">
</p>

**Metient** (**MET**astasis + gradi**ENT**) is a tool for inferring the metastatic migrations of a patient's cancer. You can find our paper published in [Nature Methods](https://www.nature.com/articles/s41592-025-02924-8).


## Installation

```bash
mamba create -n met -c conda-forge python=3.9
mamba activate met
mamba install -c conda-forge graphviz pygraphviz ipython
pip install metient
```

> [!TIP]
> If `pip install metient` fails with `fatal error: graphviz/cgraph.h: No such file or directory`, set environment variables to point to your graphviz headers:
> ```bash
> export CFLAGS="-I/path/to/mamba/env/include"
> export LDFLAGS="-L/path/to/mamba/env/lib"
> pip install pygraphviz --user
> pip install metient
> ```

## Quickstart

Run Metient on a single patient with pre-calibrated weights:

```python
import metient as met

weights = met.Weights.pancancer_genetic_organotropism_uniform_weighting()
print_config = met.PrintConfig(visualize=True, save_outputs=True, k_best_trees=5)

results = met.evaluate(
    tree_fn="path/to/tree.txt",
    tsv_fn="path/to/mutations.tsv",
    weights=weights,
    print_config=print_config,
    output_dir="./output",
    run_name="patient_1"
)
```

If you have a cohort and want Metient to learn the best weights from your data:

```python
import metient as met

print_config = met.PrintConfig(visualize=True, save_outputs=True, k_best_trees=5)

results = met.calibrate(
    tree_fns=["patient1_tree.txt", "patient2_tree.txt", "patient3_tree.txt"],
    tsv_fns=["patient1.tsv", "patient2.tsv", "patient3.tsv"],
    print_config=print_config,
    output_dir="./output",
    run_names=["patient_1", "patient_2", "patient_3"],
    calibration_type="both"  # "genetic", "organotropism", or "both"
)
```

See the [guide](docs/guide.md#step-2-prepare-your-input-files) for input file formats.

## How should I use Metient?

Metient needs to know which clones are present at which anatomical sites. You can either give it that information directly (binary present/absent per clone per site), or give it ref/var read counts and let Metient estimate clone presence for you.

|  | **I have ref/var read counts** and want Metient to estimate clone presence | **I already know which clones are present** at each site (binary present/absent) |
|--|---|---|
| **Evaluate** (pre-set weights, any # of patients) | `met.evaluate()` — [Tutorial 3](tutorial/3_evaluate_infer_observed_clones_label_clone_tree_tutorial.ipynb) | `met.evaluate_label_clone_tree()` — [Tutorial 4](tutorial/4_evaluate_label_clone_tree_tutorial.ipynb) |
| **Calibrate** (learn weights, cohort of ~5+) | `met.calibrate()` — [Tutorial 1](tutorial/1_calibrate_infer_observed_clones_label_clone_tree_tutorial.ipynb) | `met.calibrate_label_clone_tree()` — [Tutorial 2](tutorial/2_calibrate_label_clone_tree_tutorial.ipynb) |

To run a tutorial: `git clone git@github.com:morrislab/metient.git && cd metient/tutorial/`

> **📖 [Full Guide](docs/guide.md)** — covers input/output formats, weight configuration, calibration modes, performance tuning, and analysis utilities. We recommend starting with a tutorial and using the guide as a reference.

## System requirements
Metient compute requirements depend on the input size of the data. Inputs with less than ~50 tree nodes and 6 tumor sites can be run on any computer with sufficient RAM. Inputs with larger tree sizes or tumor sites should use a GPU along with a larger amount of CPU RAM. No extra configuration is needed to run Metient on GPU (Metient will automatically detect and use a GPU if one is available).

Metient is much faster on GPU, so **for large trees, we highly recommend using a GPU**. 

Tested on macOS Sonoma (14.4) and CentOS Linux 7 (Core).

## Reproducibility

To reproduce results from our paper, please see here for input data and outputs from Metient for all datasets: https://github.com/divyakoyy/metient_reproducibility/

Notebooks that use the outputs from above to generate figures for the paper can be found here: https://github.com/morrislab/metient/tree/main/notebooks

## Questions

Please email any questions you have to divyakoyy@gmail.com, or open a GitHub issue!

## Changelog
**v1.0.0**
- Improved CPU memory footprint for large trees.
- `num_mutations` column is now optional in clone-presence TSV input.

Note: version numbering reset to semantic versioning from this release.

**v0.1.3.5.8**
- Removed pseudocount in normalization of genetic distance and organotropism scoring when averaging over migration edges. This may produce minor reranking in some cases where the migration numbers of solutions are very close compared to previous versions.

**v0.1.3.5.4**
- Fixed a bug in phyleticity classification that produced incorrect results for certain edge cases. We recommend re-running analyses if phyleticity classifications are central to your conclusions.

## Citation
If you use Metient, please cite our paper:
```
Koyyalagunta, D., Ganesh, K. & Morris, Q. Inferring cancer type-specific patterns of metastatic spread using Metient. Nat Methods 23, 574–584 (2026). https://doi.org/10.1038/s41592-025-02924-8
```