# Disclaimer: This work is primarily a small extension of the work by:
```
@inproceedings{zheng2018dags,
    author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
    year = {2018}
}
```
# This Repo / Our Analysis 
1. **Weight Range Analysis**: This directory contains the code and results from running the NOTEARS linear algorithm with different weight ranges and our bootstrap aggregation (bagging) implementation. We investigate how different weight initializations affect the structure learning performance.

2. **Cytometry Analysis**: This directory contains code and results from running the bootstrap implementation of NOTEARS linear algorithm on the dataset provided by Sachs et al. (2005). This is the same dataset used in the original NOTEARS paper for their real-world benchmarks and is widely used in the biological research community for causal discovery.

3. **Visualization**: The `visualize_results.py` script generates plots to compare the performance metrics (FDR, TPR, FPR, SHD) between standard NOTEARS and our bootstrap implementation across different experimental settings.

4. **Scientific Computing Cluster Implementation**
-  This directory contains two shell scripts (`slurm_weight_ranges.sh`, `slurm_cyto_analysis.sh`) which can be used to execute the experiments on the Scientific Computing Cluster at the University of Leipzig
- The scripts are configured to automatically generate visualizations after successful execution 


