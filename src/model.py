import torch
import torch.nn as nn
import torch.nn.functional as F

class EcClassifier(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=128, dropout_rate=0.1):
        """
        初始化 EC 分类器/投影头
        
        Args:
            input_dim (int): 输入维度 (ESM embedding size, default 1280)
            hidden_dim (int): 隐藏层维度 (default 512)
            output_dim (int): 输出维度 (Projection size, default 128)
            dropout_rate (float): Dropout 比率 (default 0.1)
        """
        super().__init__()
        
        # 定义 MLP 投影头 (Projection Head)
        # 结构: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
        self.projector = nn.Sequential(
            # 第一层：降维 + 变换
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 关键：稳定特征分布，防止过拟合
            nn.ReLU(),                 # 关键：引入非线性
            nn.Dropout(dropout_rate),  # 关键：防止对特定神经元过拟合
            
            # 第二层：映射到最终的度量空间
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，有助于保持梯度稳定"""
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
        前向传播
        Args:
            x (torch.Tensor): 输入特征 [batch_size, input_dim]
        Returns:
            z (torch.Tensor): 归一化后的嵌入 [batch_size, output_dim]
        """
        # 1. 通过 MLP 投影头
        z = self.projector(x)
        
        # 2. L2 归一化 (对于余弦相似度至关重要)
        # 将向量长度缩放为 1，这样点积就等于余弦相似度
        return F.normalize(z, p=2, dim=-1)