import torch
import torch.nn as nn

class AsymmetricHuberLoss(nn.Module):
    def __init__(self, delta=150.0, omega=0.85):
        super().__init__()
        self.delta, self.omega = delta, omega

    def forward(self, y_pred, y_true):
        err = y_true - y_pred
        abs_err = torch.abs(err)
        quadratic = torch.clamp(abs_err, max=self.delta)
        linear = abs_err - quadratic
        huber_loss = 0.5 * quadratic.pow(2) + self.delta * linear
        
        weight = torch.where(err > 0, torch.full_like(err, self.omega), torch.full_like(err, 1.0 - self.omega))
        return torch.mean(weight * huber_loss)