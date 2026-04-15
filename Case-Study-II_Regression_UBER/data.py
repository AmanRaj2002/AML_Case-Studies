import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def prepare_data(num_samples=20000, num_features=40, embed_dim=8):
    """
    Simulates route-level telemetry data.
    """
    np.random.seed(42)
    # torch.manual_step(42)

    # X represents continuous and categorical features
    # Shape: (Batch, K features, d embedding dimension)
    X = torch.randn(num_samples, num_features, embed_dim)

    # Baseline ETA provided by the physical routing engine (seconds)
    baseline_eta = torch.empty(num_samples, 1).uniform_(300, 3600)

    # BUG FIX: Remove keepdim=True and view/reshape it explicitly to (num_samples, 1)
    # to prevent 3D to 2D broadcasting inflation.
    hidden_influence = torch.sum(X, dim=(1, 2)).view(num_samples, 1) * 3.0
    true_residual = hidden_influence + torch.randn(num_samples, 1) * 15
    
    # Ground truth actual arrival time
    actual_time = baseline_eta + true_residual

    # Train/Validation Split (80/20)
    split_idx = int(0.8 * num_samples)
    
    train_dataset = TensorDataset(X[:split_idx], baseline_eta[:split_idx], actual_time[:split_idx])
    val_dataset = TensorDataset(X[split_idx:], baseline_eta[split_idx:], actual_time[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    return train_loader, val_loader