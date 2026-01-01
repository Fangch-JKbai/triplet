# triplet/hard/dataset.py
"""
数据集 - 难负样本学习
"""
import torch
from torch.utils.data import Dataset
from functools import lru_cache
import os

# ============================================================
# 模块级缓存函数（避免 lru_cache 在实例方法上的问题）
# ============================================================
@lru_cache(maxsize=50000)
def _load_embedding_cached(embedding_dir, seq_id):
    """
    模块级缓存函数
    
    这样多个 Dataset 实例可以共享缓存，避免内存泄漏
    """
    emb_path = os.path.join(embedding_dir, f"{seq_id}.pt")
    data = torch.load(emb_path, map_location='cpu', weights_only=True)
    return data['mean_representations'].float()


def clear_embedding_cache():
    """清理 embedding 缓存（如需要时调用）"""
    _load_embedding_cached.cache_clear()


class TripletDataset(Dataset):
    """
    Triplet学习数据集
    
    特点：
    1. 直接加载ESM embedding（不经过模型）
    2. 返回原始1280维向量
    3. 支持多标签EC号
    """
    
    def __init__(self, seq_ids, id_to_ecs_map, embedding_dir):
        """
        Args:
            seq_ids: 序列ID列表
            id_to_ecs_map: ID到EC号列表的映射
            embedding_dir: embedding文件目录
        """
        self.seq_ids = seq_ids
        self.id_to_ecs_map = id_to_ecs_map
        self.embedding_dir = embedding_dir
        
        # 过滤掉没有embedding的样本
        self.valid_ids = []
        for seq_id in seq_ids:
            emb_path = os.path.join(embedding_dir, f"{seq_id}.pt")
            if os.path.exists(emb_path):
                self.valid_ids.append(seq_id)
        
        print(f"Dataset initialized: {len(self.valid_ids)}/{len(seq_ids)} samples have embeddings")
        
        # 构建EC到索引的映射（用于采样器）
        self.ec_to_indices = {}
        for idx, seq_id in enumerate(self.valid_ids):
            ec_list = id_to_ecs_map.get(seq_id, [])
            for ec in ec_list:
                if ec not in self.ec_to_indices:
                    self.ec_to_indices[ec] = []
                self.ec_to_indices[ec].append(idx)
        
        print(f"Found {len(self.ec_to_indices)} unique EC numbers")
    
    def __len__(self):
        return len(self.valid_ids)
    
    def _load_embedding(self, seq_id):
        """加载单个embedding（使用模块级缓存）"""
        return _load_embedding_cached(self.embedding_dir, seq_id)
    
    def __getitem__(self, idx):
        seq_id = self.valid_ids[idx]
        embedding = self._load_embedding(seq_id)
        ec_list = self.id_to_ecs_map.get(seq_id, [])
        
        # 取第一个EC号作为主标签（用于采样）
        primary_ec = sorted(ec_list)[0] if ec_list else "unknown"
        
        return {
            "seq_id": seq_id,
            "embedding": embedding,
            "primary_ec": primary_ec,
            "all_ecs": ec_list
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    embeddings = torch.stack([item["embedding"] for item in batch])
    seq_ids = [item["seq_id"] for item in batch]
    primary_ecs = [item["primary_ec"] for item in batch]
    all_ecs = [item["all_ecs"] for item in batch]
    
    return {
        "embeddings": embeddings,
        "seq_ids": seq_ids,
        "primary_ecs": primary_ecs,
        "all_ecs": all_ecs
    }