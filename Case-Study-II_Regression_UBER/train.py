import torch.optim as optim
from model import RouteResidualNet
from loss import AsymmetricHuberLoss

def train_model(train_loader, epochs=8):
    model = RouteResidualNet()
    criterion = AsymmetricHuberLoss(delta=150.0, omega=0.85)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("--- Initiating Training Pipeline ---")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, baseline_eta, actual_time in train_loader:
            optimizer.zero_grad()
            
            pred_residual = model(features)
            pred_time = baseline_eta + pred_residual
            
            loss = criterion(pred_time, actual_time)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Asymmetric Huber Loss: {avg_loss:.4f}")
        
    return model