import torch
import torch.nn as nn

class AsymmetricHuberLoss(nn.Module):
    """
    Custom loss function aligning algorithm with business reality.
    """
    def __init__(self, delta=120.0, omega=0.85):
        super().__init__()
        self.delta = delta
        self.omega = omega

    def forward(self, y_pred, y_true):
        err = y_true - y_pred
        abs_err = torch.abs(err)
        
        quadratic = torch.clamp(abs_err, max=self.delta)
        linear = abs_err - quadratic
        huber_loss = 0.5 * quadratic.pow(2) + self.delta * linear
        
        # ERROR FIX: Cast scalar float values to tensors matching the shape/device of 'err'
        weight_under = torch.full_like(err, self.omega)
        weight_over = torch.full_like(err, 1.0 - self.omega)
        weight = torch.where(err > 0, weight_under, weight_over)
        
        return torch.mean(weight * huber_loss)