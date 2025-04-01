import os
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Define paths for logging and outputs
RESULTS_DIR = Path("results")
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrices"
LOSS_CURVE_DIR = RESULTS_DIR / "loss_curves"

# Ensure directories exist
CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
LOSS_CURVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "experiment_logs.csv"

def save_confusion_matrix(true_labels, predicted_labels, class_names, class_name, run_name):
    """
    Save a confusion matrix for a specific class as a heatmap with a dynamic filename.
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {class_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # Use run_name in the filename
    output_path = CONFUSION_MATRIX_DIR / f"{run_name}_{class_name}_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")
    
    return str(output_path)  




# Save loss curve
def save_loss_curve(train_losses, val_losses, run_name):
    """
    Save the training and validation loss curves as a PNG file with a dynamic filename.
    Returns the file path for logging.
    """
    if not train_losses or not val_losses:
        print("⚠️ Warning: Empty loss lists received. Skipping loss curve saving.")
        return 

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Use run_name in the filename
    output_path = LOSS_CURVE_DIR / f"{run_name}_loss_curve.png"
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Loss curve saved to {output_path}")
    
    return str(output_path) 





# Log experiment results to CSV
def log_experiment_results(log_file, experiment_details):
    """
    Logs experiment details to a CSV file.
    """
    with log_file.open(mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=experiment_details.keys())
        if log_file.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(experiment_details)