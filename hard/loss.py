# triplet/hard/loss.py
"""
难负样本学习的损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class AdaptiveTripletLoss(nn.Module):
    """
    自适应Margin的Triplet Loss
    
    核心思想：
    1. 对每个anchor，在batch内找最难的负样本
    2. margin根据anchor-positive和anchor-negative的EC距离动态调整
    3. 距离近的EC对要求更大的margin
    """
    
    def __init__(
        self,
        margin_base=0.5,
        margin_scale=0.3,
        distance_dict_path=None,
        distance_aware=True
    ):
        """
        Args:
            margin_base: 基础margin
            margin_scale: margin调整幅度
            distance_dict_path: EC距离字典路径
            distance_aware: 是否根据EC距离调整margin
        """
        super().__init__()
        self.margin_base = margin_base
        self.margin_scale = margin_scale
        self.distance_aware = distance_aware
        
        # 加载EC距离矩阵
        if distance_aware and distance_dict_path:
            with open(distance_dict_path, 'rb') as f:
                self.distance_dict = pickle.load(f)
        else:
            self.distance_dict = None
    
    def _get_ec_distance(self, ec1, ec2):
        """获取两个EC号之间的距离"""
        if self.distance_dict is None:
            return 0.5  # 默认中等距离
        
        if ec1 in self.distance_dict and ec2 in self.distance_dict[ec1]:
            return self.distance_dict[ec1][ec2]
        elif ec2 in self.distance_dict and ec1 in self.distance_dict[ec2]:
            return self.distance_dict[ec2][ec1]
        else:
            return 0.5
    
    def _build_ec_index_mapping(self, primary_ecs):
        """构建 EC 到索引的映射，避免 hash 碰撞"""
        unique_ecs = list(set(primary_ecs))
        ec_to_idx = {ec: i for i, ec in enumerate(unique_ecs)}
        return ec_to_idx, unique_ecs
    
    def forward(self, embeddings, primary_ecs):
        """
        Args:
            embeddings: [batch_size, embed_dim] 已归一化的embedding
            primary_ecs: [batch_size] 每个样本的主EC号
        
        Returns:
            loss: scalar
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(embeddings, embeddings.t())  # [B, B]
        
        # 构建 EC 索引映射（避免 hash 碰撞）
        ec_to_idx, unique_ecs = self._build_ec_index_mapping(primary_ecs)
        ec_indices = torch.tensor([ec_to_idx[ec] for ec in primary_ecs], device=device)
        
        # 构建正负样本 mask（不使用 inplace 操作）
        same_ec = (ec_indices.unsqueeze(0) == ec_indices.unsqueeze(1))  # [B, B]
        
        # 对角线 mask
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # 正样本 mask: 同 EC 且不是自己
        pos_mask = same_ec & (~diag_mask)
        
        # 负样本 mask: 不同 EC
        neg_mask = (~same_ec) & (~diag_mask)
        
        # 检查是否有有效的正负样本对
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 对每个 anchor 找 hardest negative（相似度最高的负样本）
        # 使用 torch.where 避免 inplace 操作
        neg_sim = torch.where(neg_mask, similarity_matrix, 
                              torch.tensor(-float('inf'), device=device))
        hardest_neg_sim, hardest_neg_indices = neg_sim.max(dim=1)  # [B]
        
        # 计算 adaptive margin（如果启用）
        if self.distance_aware and self.distance_dict is not None:
            margins = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                neg_idx = hardest_neg_indices[i].item()
                ec_dist = self._get_ec_distance(primary_ecs[i], primary_ecs[neg_idx])
                # 距离越近，margin越大（更难区分）
                margins[i] = self.margin_base + self.margin_scale * (1 - ec_dist)
        else:
            margins = torch.full((batch_size,), self.margin_base, device=device)
        
        # 计算 triplet loss
        # loss = max(0, margin - pos_sim + neg_sim)
        # 使用 torch.where 避免 inplace 操作
        
        # 广播 margin 和 hardest_neg_sim
        # loss_matrix[i,j] = max(0, margin[i] - sim[i,j] + hardest_neg_sim[i])
        raw_loss = margins.unsqueeze(1) - similarity_matrix + hardest_neg_sim.unsqueeze(1)
        loss_matrix = F.relu(raw_loss)
        
        # 只计算正样本对的 loss（使用乘法代替 inplace 赋值）
        loss_matrix = loss_matrix * pos_mask.float()
        
        num_valid = pos_mask.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss_matrix.sum() / num_valid


class CircleLoss(nn.Module):
    """
    Circle Loss - 对难负样本特别有效
    
    论文: Circle Loss: A Unified Perspective of Pair Similarity Optimization
    """
    
    def __init__(self, m=0.25, gamma=80):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
    
    def forward(self, embeddings, primary_ecs):
        """
        Args:
            embeddings: [batch_size, embed_dim]
            primary_ecs: [batch_size]
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(embeddings, embeddings.t())
        
        # 构建 EC 索引映射（避免 hash 碰撞）
        unique_ecs = list(set(primary_ecs))
        ec_to_idx = {ec: i for i, ec in enumerate(unique_ecs)}
        ec_indices = torch.tensor([ec_to_idx[ec] for ec in primary_ecs], device=device)
        
        # 构建 mask（不使用 inplace 操作）
        same_ec = (ec_indices.unsqueeze(0) == ec_indices.unsqueeze(1))
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # 正样本 mask: 同 EC 且不是自己
        pos_mask = same_ec & (~diag_mask)
        # 负样本 mask: 不同 EC
        neg_mask = (~same_ec) & (~diag_mask)
        
        # 提取正负样本相似度
        sp = similarity_matrix[pos_mask]
        sn = similarity_matrix[neg_mask]
        
        if sp.numel() == 0 or sn.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Circle Loss公式
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)
        
        delta_p = 1 - self.m
        delta_n = self.m
        
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        
        return loss