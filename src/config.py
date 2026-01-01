import torch
from datetime import datetime
import numpy as np

SEED = 2025
N_SPLITS = 5
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# ============ 实验模式配置 ============
EXPERIMENT_MODE = "final_training"

ARTIFACTS_CONFIG = {
    "output_dir": "experiments",
    "run_name": "cpt_sub"
}

# === 修改点 1: 对齐新的 MLP 模型参数 ===
MODEL_CONFIG = {
    'input_dim': 1280,
    'hidden_dim': 512,
    'output_dim': 128,
    'dropout_rate': 0.0,
}

TRAIN_CONFIG = {
    "P": 16,
    "K": 16,
    "temperature": 0.1,
    "epochs": 2000,
    "learning_rate": 5e-5,
    "scheduler_type": "cosine_warmup",
    "warmup_epochs": 200,
    "grad_clip_norm": 0,
    "patience": 20,
    "scheduler_eta_min": 1e-8,
}

embedding_paths = {
    "esm":"/home/fangchh/workdir/triplet/data/ESM_2/protein_embeddings",
    "cpt":"/home/fangchh/workdir/triplet/data/ESM_CPT/protein_embeddings",
    "sub":"/home/fangchh/workdir/triplet/data/ESM_SUB/protein_embeddings",
    "cpt_sub":"/home/fangchh/workdir/triplet/data/ESM_CPT_SUB/protein_embeddings"
}

DATA_PATHS = {
    "csv_path": "/home/fangchh/workdir/triplet/data/split100.csv",
    "train_csv_path":"train_split_70.csv",
    "val_csv_path":"val_split_70.csv",
    "embedding_dir": embedding_paths["cpt_sub"],
    "external_test_sets": {
        "price": "/home/fangchh/workdir/triplet/data/price.csv",
        "new": "/home/fangchh/workdir/triplet/data/new.csv"
    }
}

DATA_CONFIG = {
    'val_split_ratio': 0.2,
    'stratify_level': 3,
}

EVAL_CONFIG = {
    "weighted": True,
    "weighting_mode": "softmax",          # 强烈建议明确写上（你目前默认就是 softmax）
    "normalize_embeddings": True,
    "keep_gallery_on_device": True,
    "sim_dtype_fp32": True,
    "default_k": 3,
    "default_tau": 0.35,
    "adaptive_tau": False,                 # 开启动态 tau
    "adaptive_tau_mode": "entropy",       # 方案C
    "tau_min": 0.15,                   # OOD 友好：更容易预测出结果（Recall↑）
    "tau_max": 0.35,                      # ID 保守：避免乱报（Precision↑）
    "tau_mix": 0.0,                      # 0=纯动态；0.35 表示 35% 用固定default_tau稳住
    "entropy_clamp_negative": True,       # 防 raw/负相似度导致熵异常（softmax模式也无害）
}


TUNE_CONFIG = {
    # ============ 用于 eval_hyperparam_search 模式 ============
    'k_search': [3],
    'tau_search_range': (0.35, 0.35, 1)
}

DATALOADER_CONFIG = {
    "num_workers": 16,
    "pin_memory": True,
    "persistent_workers": True if torch.cuda.is_available() else False, 
    "prefetch_factor": 2,
}

# ============ 超参数搜索配置 ============
# 用于 train_hyperparam_search 模式
HYPERPARAM_SEARCH_SPACE = {
    'learning_rate': [1e-3, 1e-4],
    'temperature': [0.1, 0.5],
    'patience': [10, 20],
    'P': [16],
    'K': [4],
    'dropout_rate': [0.1, 0.2],
}