#!/bin/bash
#SBATCH --job-name=notears_bootstrap
#SBATCH --output=notears.out
#SBATCH --error=notears.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=paul  # or another suitable partition

# Load Python module
module load Python/3.9

# Activate virtual environment (assuming it already exists)
source /home/sc.uni-leipzig.de/og98ohex/notears/notears/venv_notears/bin/activate  # Absolute path!

# Test if required packages are available
python -c "import numpy, scipy, pandas; print('Packages OK')"

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Execute Python script
python weight_range_analysis/main.py

# Generate visualizations if experiment completed successfully
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully. Generating visualizations..."
    python weight_range_analysis/visualize_results.py
    echo "Visualizations completed at $(date)"
else
    echo "Experiment failed. Skipping visualization."
fi

echo "Job completed at $(date)"