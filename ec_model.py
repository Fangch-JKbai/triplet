import torch
import esm
import torch.nn as nn
import torch.nn.functional as F
import os


class ec_classifier(nn.Module):
    def __init__(self, input_dim, d1=512, d2=256, d_z=128, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d1)
        self.ln1 = nn.LayerNorm(d1)
        self.fc2 = nn.Linear(d1, d2)
        self.ln2 = nn.LayerNorm(d2)
        self.fc3 = nn.Linear(d2, d_z)
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化更稳：Xavier + 零偏置
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = self.dropout(h)
        z = self.fc3(h)
        return F.normalize(z, p=2, dim=-1)