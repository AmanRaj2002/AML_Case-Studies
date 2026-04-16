import torch
import torch.optim as optim
import numpy as np

def fit(model, train_dl, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for features, baseline, actual in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(features, baseline), actual)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dl)

def score(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_errors = []
    with torch.no_grad():
        for features, baseline, actual in dataloader:
            pred_time = model(features, baseline)
            total_loss += criterion(pred_time, actual).item()
            all_errors.extend((actual - pred_time).numpy().flatten())
    return total_loss / len(dataloader), np.array(all_errors)

def evaluate(errors, model_name="Model"):
    mae = np.mean(np.abs(errors))
    under_ratio = np.sum(errors > 0) / len(errors)
    print(f"--- Evaluation: {model_name} ---")
    print(f"MAE: {mae:.2f} seconds")
    print(f"Under-prediction Ratio: {under_ratio:.2%} (Optimized for < 50%)")
    return mae, under_ratio

def validate(model, model_name, train_dl, val_dl, criterion, epochs=8, lr=0.005):
    print(f"\nTraining {model_name}...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_history, val_history = [], []
    
    for epoch in range(epochs):
        train_loss = fit(model, train_dl, criterion, optimizer)
        val_loss, _ = score(model, val_dl, criterion)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")
        
    _, val_errors = score(model, val_dl, criterion)
    evaluate(val_errors, f"{model_name} (Validation Set)")
    return train_history, val_history, val_errors