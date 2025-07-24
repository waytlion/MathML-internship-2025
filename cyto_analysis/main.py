import numpy as np
import pandas as pd
import os
import time
import sys
from datetime import timedelta
from utils import notears_linear, count_accuracy, TeeOutput, setup_logging

# Load the cytometry datasets
def load_cyto_data():
    """Load cytometry data and target files into NumPy arrays"""
    
     # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If we're in the main directory, look in the cyto_analysis subdirectory
    if os.path.basename(script_dir) != 'cyto_analysis':
        data_dir = os.path.join(script_dir, 'cyto_analysis')
    else:
        data_dir = script_dir
    
    # Construct full paths to the CSV files
    data_path = os.path.join(data_dir, 'input_data/cyto_full_data.csv')
    target_path = os.path.join(data_dir, 'input_data/cyto_full_target.csv')
    
    # Check if files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target file not found: {target_path}")
    
    # Load the data file
    data_df = pd.read_csv(data_path)
    # Load the target file
    target_df = pd.read_csv(target_path)

    protein_names = list(data_df.columns)
    adjacency_df = pd.DataFrame(0, index = protein_names, columns=protein_names)
    for _, row in target_df.iterrows():
        cause = row.iloc[0]
        effect = row.iloc[1]
        adjacency_df.at[cause, effect] = 1

    # Convert to NumPy arrays
    X = data_df.values
    adjacency_matrix = adjacency_df.values
    
    return X, adjacency_matrix

if __name__ == '__main__':    
    # Set up logging to redirect output to file
    log_file_path = setup_logging()
    print(f"Log file created at: {log_file_path}")
    
    start_time_total  = time.time()
    np.random.seed(0)
    bootstrap_samples = 20
    n_rows = 100
    w_threshold = 0.1
    
    # Create results directory with absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detailed_results_dir = os.path.join(script_dir, 'detailed_results')
    os.makedirs(detailed_results_dir, exist_ok=True)
    print(f"Results will be saved to: {detailed_results_dir}")

    ### Load the datasets
    X, B_true = load_cyto_data()
    print("Original dataset shape:", X.shape)
    
    # Save the ground truth adjacency matrix (B_true) to CSV
    np.savetxt(os.path.join(detailed_results_dir, "B_true.csv"), B_true, delimiter=',', fmt='%d')

    # Randomly sample n-rows rows
    selected_rows = np.random.choice(X.shape[0], n_rows, replace=False)
    X = X[selected_rows, :]    
    print(f"Adjacency matrix: {B_true.shape}")
    print(" dataset shape after subsampling:", X.shape)

    ### Estimate DAG 
    data_DAG = X.copy()
    # Standardize Features
    data_DAG = data_DAG - np.mean(data_DAG, axis=0, keepdims=True)
    data_DAG = data_DAG / np.std(data_DAG, axis=0, keepdims=True)
    W_est = notears_linear(data_DAG, lambda1=0.1, loss_type='l2')

    ### Accuracies: compare stat with true (compare with weights set to 1 and without set to 1)    # Calculate accuracies for continuous W_est
    acc_w_cont = count_accuracy(B_true, W_est != 0)
    
    # Create binary version of W_est (set non-zero values to 1)
    W_est_binary = W_est.copy()
    W_est_binary[W_est_binary != 0] = 1
    
    # Calculate accuracies for binary W_est
    acc_w_binary = count_accuracy(B_true, W_est_binary != 0)

    # Method 2: Bootstrap estimation
    W_est_bootstrapped = []
    for i in range(bootstrap_samples):
        # Create bootstrap sample by sampling with replacement
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        subsample = X[indices]
        # Standardize Features
        subsample = subsample - np.mean(subsample, axis=0, keepdims=True)
        subsample = subsample / np.std(subsample, axis=0, keepdims=True)
        # Run NOTEARS on bootstrap sample
        print(f"Running bootstrap iteration {i+1}/{bootstrap_samples}")
        W_est_bootstrapped.append(notears_linear(subsample, lambda1=0.1, loss_type="l2"))
    
    # Stack and average
    W_stack = np.stack(W_est_bootstrapped)
    W_mean = np.mean(W_stack, axis=0)
    W_mean[np.abs(W_mean) < w_threshold] = 0


    # Evaluate bootstrapped model
    acc_with_bootstrap = count_accuracy(B_true, W_mean != 0)

    ### Print Weight Matrices
    # Print weight statistics for W_est (non-zero values only)
    w_nonzero = W_est[W_est != 0]
    if len(w_nonzero) > 0:  # Check if there are any non-zero weights
        w_min = np.min(w_nonzero).round(2)
        w_max = np.max(w_nonzero).round(2)
        w_mean = np.mean(w_nonzero).round(2)
        w_median = np.median(w_nonzero).round(2)
        print(f"\n===== Weight Statistics for W_est (non-zero values) =====")
        print(f"Min: {w_min}, Max: {w_max}, Mean: {w_mean}, Median: {w_median}")
    
    print(f"\n===== Estimated Weight Matrix continous values (W_est) =====")
    print(np.round(W_est, 2))
    print(f"\n===== Estimated Weight Matrix Binary mapped values (W_est_binary) =====")
    print(W_est_binary)
    print(f"\n===== Estimated Weight Matrix Bootstrapped (W_est_bootstrapped) =====")
    print(np.round(W_mean, 2))
    
    # Print Accuracies with rounded values
    print(f"Accuracies W_Est continuous: {', '.join([f'{k}: {v:.2f}' for k, v in acc_w_cont.items()])}")
    print(f"Accuracies W_est_binary: {', '.join([f'{k}: {v:.2f}' for k, v in acc_w_binary.items()])}")
    print(f"Accuracies W_Est bootstrapped: {', '.join([f'{k}: {v:.2f}' for k, v in acc_with_bootstrap.items()])}")

    # Save weight matrices with values rounded to 2 decimal places
    np.savetxt(os.path.join(detailed_results_dir, "W_est_binary.csv"), W_est_binary, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(detailed_results_dir, "W_est_bootstrapped.csv"), np.round(W_mean, 2), delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(detailed_results_dir, "W_est_continuous.csv"), np.round(W_est, 2), delimiter=',', fmt='%.2f')

    # Save accuracies as CSV files with values rounded to 2 decimal places
    pd.DataFrame([{k: round(v, 2) for k, v in acc_w_cont.items()}]).to_csv(os.path.join(detailed_results_dir, "accuracies_continuous.csv"), index=False)
    pd.DataFrame([{k: round(v, 2) for k, v in acc_w_binary.items()}]).to_csv(os.path.join(detailed_results_dir, "accuracies_binary.csv"), index=False)
    pd.DataFrame([{k: round(v, 2) for k, v in acc_with_bootstrap.items()}]).to_csv(os.path.join(detailed_results_dir, "accuracies_bootstrapped.csv"), index=False)
    ### Total Time 
    total_time = time.time() - start_time_total
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"Total execution time: {total_time_str} (format: H:MM:SS)")
    
    # Restore original stdout and close the log file
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
import pandas as pd
import os
import time
import sys
from datetime import timedelta
from utils import notears_linear, count_accuracy, TeeOutput, setup_logging

# Load the cytometry datasets
def load_cyto_data():
    """Load cytometry data and target files into NumPy arrays"""
    
     # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If we're in the main directory, look in the cyto_analysis subdirectory
    if os.path.basename(script_dir) != 'cyto_analysis':
        data_dir = os.path.join(script_dir, 'cyto_analysis')
    else:
        data_dir = script_dir
    
    # Construct full paths to the CSV files
    data_path = os.path.join(data_dir, 'input_data/cyto_full_data.csv')
    target_path = os.path.join(data_dir, 'input_data/cyto_full_target.csv')
    
    # Check if files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target file not found: {target_path}")
    
    # Load the data file
    data_df = pd.read_csv(data_path)
    # Load the target file
    target_df = pd.read_csv(target_path)

    protein_names = list(data_df.columns)
    adjacency_df = pd.DataFrame(0, index = protein_names, columns=protein_names)
    for _, row in target_df.iterrows():
        cause = row.iloc[0]
        effect = row.iloc[1]
        adjacency_df.at[cause, effect] = 1

    # Convert to NumPy arrays
    X = data_df.values
    adjacency_matrix = adjacency_df.values
    
    return X, adjacency_matrix

if __name__ == '__main__':    
    # Set up logging to redirect output to file
    log_file_path = setup_logging()
    print(f"Log file created at: {log_file_path}")
    
    start_time_total  = time.time()
    np.random.seed(0)
    bootstrap_samples = 20
    n_rows = 100
    w_threshold = 0.1
    
    # Create results directory with absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detailed_results_dir = os.path.join(script_dir, 'detailed_results')
    os.makedirs(detailed_results_dir, exist_ok=True)
    print(f"Results will be saved to: {detailed_results_dir}")

    ### Load the datasets
    X, B_true = load_cyto_data()
    print("Original dataset shape:", X.shape)
    
    # Save the ground truth adjacency matrix (B_true) to CSV
    np.savetxt(os.path.join(detailed_results_dir, "B_true.csv"), B_true, delimiter=',', fmt='%d')

    # Randomly sample n-rows rows
    selected_rows = np.random.choice(X.shape[0], n_rows, replace=False)
    X = X[selected_rows, :]    
    print(f"Adjacency matrix: {B_true.shape}")
    print(" dataset shape after subsampling:", X.shape)

    ### Estimate DAG 
    data_DAG = X.copy()
    # Standardize Features
    data_DAG = data_DAG - np.mean(data_DAG, axis=0, keepdims=True)
    data_DAG = data_DAG / np.std(data_DAG, axis=0, keepdims=True)
    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')

    ### Accuracies: compare stat with true (compare with weights set to 1 and without set to 1)    # Calculate accuracies for continuous W_est
    acc_w_cont = count_accuracy(B_true, W_est != 0)
    
    # Create binary version of W_est (set non-zero values to 1)
    W_est_binary = W_est.copy()
    W_est_binary[W_est_binary != 0] = 1
    
    # Calculate accuracies for binary W_est
    acc_w_binary = count_accuracy(B_true, W_est_binary != 0)

    # Method 2: Bootstrap estimation
    W_est_bootstrapped = []
    for i in range(bootstrap_samples):
        # Create bootstrap sample by sampling with replacement
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        subsample = X[indices]
        # Standardize Features
        subsample = subsample - np.mean(subsample, axis=0, keepdims=True)
        subsample = subsample / np.std(subsample, axis=0, keepdims=True)
        # Run NOTEARS on bootstrap sample
        print(f"Running bootstrap iteration {i+1}/{bootstrap_samples}")
        W_est_bootstrapped.append(notears_linear(subsample, lambda1=0.1, loss_type="l2"))
    
    # Stack and average
    W_stack = np.stack(W_est_bootstrapped)
    W_mean = np.mean(W_stack, axis=0)
    W_mean[np.abs(W_mean) < w_threshold] = 0


    # Evaluate bootstrapped model
    acc_with_bootstrap = count_accuracy(B_true, W_mean != 0)

    ### Print Weight Matrices
    # Print weight statistics for W_est (non-zero values only)
    w_nonzero = W_est[W_est != 0]
    if len(w_nonzero) > 0:  # Check if there are any non-zero weights
        w_min = np.min(w_nonzero).round(2)
        w_max = np.max(w_nonzero).round(2)
        w_mean = np.mean(w_nonzero).round(2)
        w_median = np.median(w_nonzero).round(2)
        print(f"\n===== Weight Statistics for W_est (non-zero values) =====")
        print(f"Min: {w_min}, Max: {w_max}, Mean: {w_mean}, Median: {w_median}")
    
    print(f"\n===== Estimated Weight Matrix continous values (W_est) =====")
    print(np.round(W_est, 2))
    print(f"\n===== Estimated Weight Matrix Binary mapped values (W_est_binary) =====")
    print(W_est_binary)
    print(f"\n===== Estimated Weight Matrix Bootstrapped (W_est_bootstrapped) =====")
    print(np.round(W_mean, 2))
    
    # Print Accuracies with rounded values
    print(f"Accuracies W_Est continuous: {', '.join([f'{k}: {v:.2f}' for k, v in acc_w_cont.items()])}")
    print(f"Accuracies W_est_binary: {', '.join([f'{k}: {v:.2f}' for k, v in acc_w_binary.items()])}")
    print(f"Accuracies W_Est bootstrapped: {', '.join([f'{k}: {v:.2f}' for k, v in acc_with_bootstrap.items()])}")

    # Save weight matrices with values rounded to 2 decimal places
    np.savetxt(os.path.join(detailed_results_dir, "W_est_binary.csv"), W_est_binary, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(detailed_results_dir, "W_est_bootstrapped.csv"), np.round(W_mean, 2), delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(detailed_results_dir, "W_est_continuous.csv"), np.round(W_est, 2), delimiter=',', fmt='%.2f')

    # Save accuracies as CSV files with values rounded to 2 decimal places
    pd.DataFrame([{k: round(v, 2) for k, v in acc_w_cont.items()}]).to_csv(os.path.join(detailed_results_dir, "accuracies_continuous.csv"), index=False)
    pd.DataFrame([{k: round(v, 2) for k, v in acc_w_binary.items()}]).to_csv(os.path.join(detailed_results_dir, "accuracies_binary.csv"), index=False)
    pd.DataFrame([{k: round(v, 2) for k, v in acc_with_bootstrap.items()}]).to_csv(os.path.join(detailed_results_dir, "accuracies_bootstrapped.csv"), index=False)
    ### Total Time 
    total_time = time.time() - start_time_total
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"Total execution time: {total_time_str} (format: H:MM:SS)")
    
    # Restore original stdout and close the log file
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__