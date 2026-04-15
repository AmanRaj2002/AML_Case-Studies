import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    """
    Linear Transformer Attention block to satisfy millisecond latency budgets.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.elu = nn.ELU() 

    def forward(self, x):
        Q = self.elu(self.q_proj(x)) + 1.0 
        K = self.elu(self.k_proj(x)) + 1.0
        V = self.v_proj(x)
        
        KV = torch.einsum('bkd,bkv->bdv', K, V)
        Z = 1.0 / (torch.einsum('bkd,bd->bk', Q, K.sum(dim=1)) + 1e-6)
        out = torch.einsum('bkd,bdv->bkv', Q, KV) * Z.unsqueeze(-1)
        return out

class RouteResidualNet(nn.Module):
    """
    Encoder-Decoder architecture for predicting route-level ETA residuals.
    """
    def __init__(self, num_features=40, embed_dim=8):
        super().__init__()
        self.attention = LinearAttention(embed_dim)
        self.fc1 = nn.Linear(num_features * embed_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        attn_out = self.attention(x)
        flat = attn_out.view(x.size(0), -1)
        hidden = self.relu(self.fc1(flat))
        residual_pred = self.fc2(hidden)
        return residual_pred