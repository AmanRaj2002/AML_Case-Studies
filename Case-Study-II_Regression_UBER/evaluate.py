import torch
from loss import AsymmetricHuberLoss

def score_and_evaluate(model, val_loader):
    model.eval()
    criterion = AsymmetricHuberLoss(delta=150.0, omega=0.85)
    
    total_mae = 0.0
    total_loss = 0.0
    under_predictions = 0
    
    print("\n--- Initiating Scoring & Evaluation Pipeline ---")
    with torch.no_grad():
        for features, baseline_eta, actual_time in val_loader:
            pred_residual = model(features)
            pred_time = baseline_eta + pred_residual
            
            loss = criterion(pred_time, actual_time)
            total_loss += loss.item()
            
            err = actual_time - pred_time
            total_mae += torch.mean(torch.abs(err)).item()
            under_predictions += torch.sum(err > 0).item()
            
    num_batches = len(val_loader)
    num_samples = len(val_loader.dataset)
    
    print("\n[ Final Evaluation Metrics ]")
    print(f"-> Validation Asymmetric Huber Loss : {total_loss / num_batches:.4f}")
    print(f"-> Mean Absolute Error (MAE)        : {total_mae / num_batches:.2f} seconds")
    print(f"-> Under-prediction Ratio           : {under_predictions / num_samples:.2%} (Optimized for < 50%)")