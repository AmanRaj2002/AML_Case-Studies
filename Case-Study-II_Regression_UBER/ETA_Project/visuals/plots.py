import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_style():
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_data_distributions(baseline_eta, true_residual):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(baseline_eta, bins=50, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribution of Baseline ETAs (Routing Engine)')
    axes[0].set_xlabel('ETA (Seconds)')
    
    sns.histplot(true_residual, bins=50, kde=True, ax=axes[1], color='orange')
    axes[1].set_title('Distribution of True Residuals (Ground Truth Error)')
    axes[1].set_xlabel('Residual (Seconds)')
    axes[1].axvline(0, color='red', linestyle='--', label='Zero Error')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_training_convergence(train_losses, val_losses):
    setup_style()
    plt.figure(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', color='teal', label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', linestyle='--', color='orange', label='Validation Loss')
    plt.title('Training and Validation Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Asymmetric Huber Loss')
    plt.legend()
    plt.show()

def plot_error_distribution(errors, title="Prediction Error Distribution"):
    setup_style()
    under_ratio = np.sum(errors > 0) / len(errors)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=60, kde=True, color='gray')
    
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.axvspan(0, max(errors), color='red', alpha=0.1, label=f'Late / Under-predicted ({under_ratio:.1%})')
    plt.axvspan(min(errors), 0, color='green', alpha=0.1, label=f'Early / Over-predicted ({(1-under_ratio):.1%})')
    
    plt.title(title)
    plt.xlabel('Error in Seconds (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()