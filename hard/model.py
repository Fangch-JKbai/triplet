# triplet/hard/model.py
"""
模型 - 复用原来的EcClassifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingProjector(nn.Module):
    """
    Embedding投影头
    
    1280维 -> 512维 -> 128维
    """
    
    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=128, dropout_rate=0.1):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, 1280]
        Returns:
            z: [batch_size, 128] L2-normalized
        """
        z = self.projector(x)
        return F.normalize(z, p=2, dim=-1)