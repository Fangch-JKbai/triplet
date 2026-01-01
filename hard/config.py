# triplet/hard/config.py (修改版)
"""
配置文件 - 难负样本学习项目
"""
import torch

# ============================================================
# 设备配置
# ============================================================
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 2025

# ============================================================
# 数据路径
# ============================================================
DATA_PATHS = {
    # 主数据文件
    "csv_path": "/home/fangchh/workdir/triplet/data/split100.csv",
    
    # ESM embedding目录
    "embedding_dir": "/home/fangchh/workdir/triplet/data/cpt/protein_embeddings",
    
    # EC距离矩阵
    "distance_dict_path": "/home/fangchh/workdir/triplet/data/cpt/reports/distance_dict.pkl",
    
    # 外部测试集（添加你的两个测试集）
    "external_test_sets": {
        "price": "/home/fangchh/workdir/triplet/data/price.csv",
        "new": "/home/fangchh/workdir/triplet/data/new.csv"
    }
}

# ============================================================
# 数据划分
# ============================================================
DATA_CONFIG = {
    "val_split_ratio": 0.2,  # 只有train/val，没有test
    "stratify_level": 4,      # EC号分层级别（前3级）
    
    # 突变序列配置
    "use_mutants": True,      # 是否使用突变序列增强训练集
}

# ============================================================
# 模型配置
# ============================================================
MODEL_CONFIG = {
    "input_dim": 1280,      # ESM embedding维度
    "hidden_dim": 512,      # 隐藏层维度
    "output_dim": 128,      # 输出embedding维度
    "dropout_rate": 0.0,
}

# ============================================================
# 难负样本采样配置
# ============================================================
SAMPLER_CONFIG = {
    "batch_size": 256,       # 每个batch的样本数
    "samples_per_ec": 8,    # 每个EC号采样多少个样本
    
    # === 修改：上来就用难样本，不做渐进式学习 ===
    "distance_percentile_start": 80,  # 从10开始（困难）
    "distance_percentile_end": 10,    # 到10结束（困难）
    "warmup_epochs": 100,               # 不做warmup，直接用难样本
    
    "samples_per_epoch": 60000,       # 每epoch 3万样本
}

# ============================================================
# 损失函数配置
# ============================================================
LOSS_CONFIG = {
    "loss_type": "adaptive_triplet",  # 'adaptive_triplet' 或 'circle'
    
    # Adaptive Triplet Loss参数
    "margin_base": 0.5,          # 基础margin
    "margin_scale": 0.3,         # margin调整幅度
    "distance_aware": True,      # 是否根据EC距离调整margin
    
    # Circle Loss参数（如果使用）
    "m": 0.25,
    "gamma": 80,
}

# ============================================================
# 训练配置
# ============================================================
TRAIN_CONFIG = {
    "epochs": 100,               # 先跑20个epoch看效果
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip_norm": 1.0,
    
    # 早停（20个epoch不会触发）
    "patience": 50,
    "early_stop_threshold_epoch": 100,
    
    # 学习率调度
    "scheduler_type": "cosine_warmup",
    "warmup_ratio": 0.1,
    "min_lr": 1e-6,
}

# ============================================================
# 评估配置
# ============================================================
EVAL_CONFIG = {
    # kNN评估
    "k_values": [3, 5, 7, 10],
    "tau_values": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    
    # 聚类质量评估
    "compute_clustering_metrics": False,
    
    # PR曲线
    "plot_pr_curves": True,
    
    # 评估频率
    "eval_every_n_epochs": 5,  # 每10个epoch评估一次
}

# ============================================================
# 输出配置
# ============================================================
OUTPUT_CONFIG = {
    "output_dir": "./outputs",
    "save_best_model": True,
    "save_checkpoints": True,
    "checkpoint_frequency": 50,
}

# ============================================================
# DataLoader配置
# ============================================================
DATALOADER_CONFIG = {
    "num_workers": 2,
    "pin_memory": True,
    "persistent_workers": True,
}