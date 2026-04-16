import torch
import torch.nn as nn

class StaticRoutingModel(nn.Module):
    def forward(self, x, baseline_eta):
        return baseline_eta

class SimpleLinearResidual(nn.Module):
    def __init__(self, input_dim=40):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x, baseline_eta):
        return baseline_eta + self.fc(x.view(x.size(0), -1))

class LinearAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.elu = nn.ELU() 

    def forward(self, x):
        Q, K, V = self.elu(self.q_proj(x)) + 1.0, self.elu(self.k_proj(x)) + 1.0, self.v_proj(x)
        KV = torch.einsum('bkd,bkv->bdv', K, V)
        Z = 1.0 / (torch.einsum('bkd,bd->bk', Q, K.sum(dim=1)) + 1e-6)
        return torch.einsum('bkd,bdv->bkv', Q, KV) * Z.unsqueeze(-1)

class AdvancedRouteTransformer(nn.Module):
    def __init__(self, num_features=5, embed_dim=8):
        super().__init__()
        self.attention = LinearAttention(embed_dim)
        self.fc1 = nn.Linear(num_features * embed_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x, baseline_eta):
        hidden = self.relu(self.fc1(self.attention(x).view(x.size(0), -1)))
        return baseline_eta + self.fc2(hidden)