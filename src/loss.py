import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比损失函数的实现.
    参考: https://arxiv.org/pdf/2004.11362.pdf
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        输入:
            features: 特征向量, shape: [bsz, n_features].
            labels: 样本标签, shape: [bsz].
        输出:
            损失值 (标量).
        """
        device = features.device
        batch_size = features.shape[0]

        # labels [bsz, 1] == labels.T [1, bsz] -> mask [bsz, bsz]
        # mask[i, j] = 1 if labels[i] == labels[j] else 0
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 归一化特征
        features = F.normalize(features, dim=1)

        # 计算所有样本对之间的余弦相似度
        # anchor_dot_contrast = (features @ features.T) / self.temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases (对角线元素)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算 log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算每个样本的平均 log-likelihood
        # mask.sum(1) 是每个样本的正样本对数量
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # 最终损失
        loss = -mean_log_prob_pos.mean()

        return loss