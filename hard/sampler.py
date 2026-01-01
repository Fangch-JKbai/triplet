# triplet/hard/sampler.py (修改版)
"""
基于EC距离的难负样本采样器
"""
import random
import pickle
import numpy as np
from torch.utils.data import Sampler
import logging

class ECDistanceAwareSampler(Sampler):
    """
    基于EC距离矩阵的智能采样器
    
    核心思想：
    1. 每个batch包含多个EC对
    2. EC对的距离根据训练进度从远到近（curriculum learning）
    3. 确保batch内有足够的困难负样本对
    """
    
    def __init__(
        self,
        ec_to_indices,
        distance_dict_path,
        batch_size=64,
        samples_per_ec=4,
        distance_percentile_start=80,
        distance_percentile_end=10,
        warmup_epochs=100,
        samples_per_epoch=30000  # ← 新增参数
    ):
        """
        Args:
            ec_to_indices: EC号到样本索引的映射
            distance_dict_path: EC距离字典路径
            batch_size: batch大小
            samples_per_ec: 每个EC采样多少个样本
            distance_percentile_start: 初始距离百分位（大=远=简单）
            distance_percentile_end: 最终距离百分位（小=近=困难）
            warmup_epochs: 过渡期epochs
            samples_per_epoch: 每个epoch采样多少个样本
        """
        self.ec_to_indices = ec_to_indices
        self.batch_size = batch_size
        self.samples_per_ec = samples_per_ec
        self.distance_percentile_start = distance_percentile_start
        self.distance_percentile_end = distance_percentile_end
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.samples_per_epoch = samples_per_epoch  # ← 保存参数
        
        # 加载EC距离矩阵
        logging.info(f"Loading EC distance matrix from {distance_dict_path}...")
        with open(distance_dict_path, 'rb') as f:
            self.distance_dict = pickle.load(f)
        
        self.ec_list = [ec for ec in ec_to_indices.keys() if len(ec_to_indices[ec]) >= samples_per_ec]
        
        # 构建所有EC对及其距离
        self._build_ec_pairs()
        
        # === 关键修改：固定每个epoch的样本数 ===
        # 原来: self.num_batches = max(100, len(self.ec_list) * 2)  # 7422个batch
        # 现在: 根据 samples_per_epoch 计算
        self.num_batches = samples_per_epoch // batch_size
        
        logging.info(f"ECDistanceAwareSampler initialized:")
        logging.info(f"  Valid ECs: {len(self.ec_list)}")
        logging.info(f"  EC pairs: {len(self.ec_pairs)}")
        logging.info(f"  Batch size: {batch_size}")
        logging.info(f"  Samples per EC: {samples_per_ec}")
        logging.info(f"  Distance percentile: {distance_percentile_start} -> {distance_percentile_end}")
        logging.info(f"  Samples per epoch: {samples_per_epoch}")  # ← 新增日志
        logging.info(f"  Batches per epoch: {self.num_batches}")
    
    def _build_ec_pairs(self):
        """构建所有EC对及其距离"""
        self.ec_pairs = []
        
        for i, ec1 in enumerate(self.ec_list):
            for ec2 in self.ec_list[i+1:]:
                dist = self._get_ec_distance(ec1, ec2)
                if dist is not None:
                    self.ec_pairs.append((ec1, ec2, dist))
        
        # 按距离排序（从近到远）
        self.ec_pairs.sort(key=lambda x: x[2])
        
        logging.info(f"  Built {len(self.ec_pairs)} EC pairs with valid distances")
    
    def _get_ec_distance(self, ec1, ec2):
        """获取两个EC号之间的距离"""
        if ec1 in self.distance_dict and ec2 in self.distance_dict[ec1]:
            return self.distance_dict[ec1][ec2]
        elif ec2 in self.distance_dict and ec1 in self.distance_dict[ec2]:
            return self.distance_dict[ec2][ec1]
        else:
            return None
    
    def set_epoch(self, epoch):
        """设置当前epoch（用于curriculum learning）"""
        self.current_epoch = epoch
        
        if epoch % 10 == 0:
            low_pct, high_pct = self._get_current_distance_range()
            logging.info(f"Epoch {epoch}: Sampling from distance percentile [{low_pct:.1f}, {high_pct:.1f}]")
    
    def _get_current_distance_range(self):
        """计算当前应该采样的距离百分位范围"""
        if self.current_epoch >= self.warmup_epochs:
            # 训练后期：只采样困难样本
            high_pct = self.distance_percentile_end
        else:
            # 训练早期：从简单到困难线性过渡
            progress = self.current_epoch / self.warmup_epochs
            high_pct = self.distance_percentile_start - (
                self.distance_percentile_start - self.distance_percentile_end
            ) * progress
        
        low_pct = max(0, high_pct - 30)  # 窗口大小30%
        return low_pct, high_pct
    
    def __iter__(self):
        """生成batch索引"""
        # 确定当前epoch要采样的EC对范围
        low_pct, high_pct = self._get_current_distance_range()
        
        n_pairs = len(self.ec_pairs)
        low_idx = int(n_pairs * low_pct / 100)
        high_idx = int(n_pairs * high_pct / 100)
        
        # 可用的EC对
        valid_pairs = self.ec_pairs[low_idx:high_idx]
        if not valid_pairs:
            valid_pairs = self.ec_pairs  # fallback
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 计算需要多少个EC对
            num_ecs_needed = self.batch_size // self.samples_per_ec
            num_pairs = num_ecs_needed // 2
            
            # 采样EC对
            sampled_pairs = random.sample(valid_pairs, min(num_pairs, len(valid_pairs)))
            
            for ec1, ec2, _ in sampled_pairs:
                # 从每个EC中采样样本
                for ec in [ec1, ec2]:
                    indices = self.ec_to_indices[ec]
                    if len(indices) >= self.samples_per_ec:
                        sampled = random.sample(indices, self.samples_per_ec)
                    else:
                        sampled = random.choices(indices, k=self.samples_per_ec)
                    batch_indices.extend(sampled)
            
            # 确保batch大小正确
            while len(batch_indices) < self.batch_size:
                # 随机补充
                random_ec = random.choice(self.ec_list)
                indices = self.ec_to_indices[random_ec]
                batch_indices.append(random.choice(indices))
            
            batch_indices = batch_indices[:self.batch_size]
            random.shuffle(batch_indices)
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches