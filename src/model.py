# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EcClassifier(nn.Module):
    def __init__(self, input_dim=1280, d1=512, d2=256, d_z=128, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d_z)
        self.ln1 = nn.LayerNorm(d_z)
        self.fc2 = nn.Linear(d1, d2)
        self.ln2 = nn.LayerNorm(d2)
        self.fc3 = nn.Linear(d2, d_z)
        self.dropout = nn.Dropout(dropout_rate)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

    def forward(self, x):

        z = self.fc1(x)
        z = self.dropout(z)
        return F.normalize(z, p=2, dim=-1)