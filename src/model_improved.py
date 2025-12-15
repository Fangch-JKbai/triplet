# model_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EcClassifier(nn.Module):
    """
    改进版分类器，支持两种架构：
    1. 'linear': 单层线性投影 (用于纯embedding质量评估)
    2. 'mlp': 多层MLP (用于完整pipeline评估)
    """
    def __init__(self, input_dim=1280, d1=512, d2=256, d_z=128, 
                 dropout_rate=0.2, architecture='mlp'):
        super().__init__()
        self.architecture = architecture
        
        if architecture == 'linear':
            # 最简单的线性投影
            self.fc = nn.Linear(input_dim, d_z)
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)
        
        elif architecture == 'mlp':
            # 完整的多层网络
            self.fc1 = nn.Linear(input_dim, d1)
            self.ln1 = nn.LayerNorm(d1)
            self.fc2 = nn.Linear(d1, d2)
            self.ln2 = nn.LayerNorm(d2)
            self.fc3 = nn.Linear(d2, d_z)
            self.dropout = nn.Dropout(dropout_rate)
            
            # Xavier初始化
            for m in [self.fc1, self.fc2, self.fc3]:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x):
        if self.architecture == 'linear':
            z = self.fc(x)
        else:  # mlp
            x = self.dropout(F.relu(self.ln1(self.fc1(x))))
            x = self.dropout(F.relu(self.ln2(self.fc2(x))))
            z = self.fc3(x)
        
        return F.normalize(z, p=2, dim=-1)
    
    def get_embeddings_before_norm(self, x):
        """返回归一化前的embeddings，用于某些分析"""
        if self.architecture == 'linear':
            return self.fc(x)
        else:
            x = self.dropout(F.relu(self.ln1(self.fc1(x))))
            x = self.dropout(F.relu(self.ln2(self.fc2(x))))
            return self.fc3(x)