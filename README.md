# Repo Owners:
- Group 4 | Aaron Zumdick | Tilman Bockhacker

# Project Report
- ./Math_and_ML_report.pdf
  
# This Repo / Our Analysis 
1. **Weight Range Analysis**: This directory contains the code and results from running the NOTEARS linear algorithm with different weight ranges and our bootstrap aggregation (bagging) implementation. We investigate how different weight initializations affect the structure learning performance.

2. **Cytometry Analysis**: This directory contains code and results from running the bootstrap implementation of NOTEARS linear algorithm on the dataset provided by Sachs et al. (2005). This is the same dataset used in the original NOTEARS paper for their real-world benchmarks and is widely used in the biological research community for causal discovery.

3. **Visualization**: The `visualize_results.py` script generates plots to compare the performance metrics (FDR, TPR, FPR, SHD) between standard NOTEARS and our bootstrap implementation across different experimental settings.

4. **Weight Cutoff Optimization**: This directory contains a precompiled Jupyter notebook to investigate the optimal value for the weight cutoff threshold. We improved upon the static threshold value of 0.3 by developing a data-driven approach for finding the optimal threshold. For this purpose, we introduced a new evaluation metric.

5. **Scientific Computing Cluster Implementation**
-  This directory contains two shell scripts (`slurm_weight_ranges.sh`, `slurm_cyto_analysis.sh`) which can be used to execute the experiments on the Scientific Computing Cluster at the University of Leipzig
- The scripts are configured to automatically generate visualizations after successful execution 


# Usage

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/waytlion/MathML-internship-2025.git
cd MathML-internship-2025
pip install -r requirements.txt
```

> **Note:** The `notears` package is installed directly from a pinned commit of the fork used by this project (see `requirements.txt`).

## Running the Weight Range Analysis

This experiment evaluates how different weight initializations affect NOTEARS structure learning, with and without bootstrap aggregation.

```bash
python weight_range_analysis/main.py
```

Results are saved to `weight_range_analysis/detailed_results/` and summary CSVs are written to the project root (`summary_no_bootstrap.csv`, `summary_with_bootstrap.csv`).

To generate visualization plots after the experiment completes:

```bash
python weight_range_analysis/visualize_results.py
```

## Running the Cytometry Analysis

This experiment applies the bootstrap NOTEARS implementation to the Sachs et al. (2005) cytometry dataset.

```bash
python cyto_analysis/main.py
```

Results (estimated weight matrices and accuracy CSVs) are saved to `cyto_analysis/detailed_results/`. A timestamped log file is also created in that directory.

## Running the Weight Cutoff Optimization

Open the precompiled Jupyter notebook to explore data-driven threshold selection:

```bash
jupyter notebook weight_cutoff_optimization/main.ipynb
```

## Running on the Scientific Computing Cluster (SLURM)

Submit the SLURM job scripts from the repository root after activating your virtual environment:

```bash
# Weight range analysis
sbatch run_on_cluster/slurm_weight_ranges.sh

# Cytometry analysis
sbatch run_on_cluster/slurm_cyto_analysis.sh
```

The weight-range script automatically runs `visualize_results.py` upon successful completion. Logs are written to `notears.out` / `notears.err`.

---

# Disclaimer: This work is an extension of the work by:
```
@inproceedings{zheng2018dags,
    author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
    year = {2018}
}
```
