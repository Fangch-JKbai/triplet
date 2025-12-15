import torch
from datetime import datetime
import numpy as np

SEED = 42
N_SPLITS = 5
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# ============ 实验模式配置 ============
# 可选值: 
EXPERIMENT_MODE = "final_training"

ARTIFACTS_CONFIG = {
    "output_dir": "experiments",
    "run_name": "cpt"
}

MODEL_CONFIG = {
    'input_dim': 1280,
    'd1': 512,
    'd2': 256,
    'd_z': 128,
    'dropout_rate': 0.2,  # 可以搜索：[0.3, 0.4, 0.5]
}

TRAIN_CONFIG = {
    "P": 16,              # 可以搜索：[8, 16, 32]
    "K": 4,               # 可以搜索：[2, 4, 8]
    "temperature": 0.1,   # 可以搜索：[0.05, 0.07, 0.1, 0.15]
    "epochs": 1000,
    "learning_rate": 1e-3,  # 可以搜索：[1e-4, 5e-4, 1e-3, 5e-3]
    "grad_clip_norm": 0,
    "patience": 15,          # 可以搜索：[3, 5, 10, 15]
    "scheduler_eta_min": 1e-6,
}

embedding_paths = {
    "esm":"/home/fangchh/workdir/triplet/data/base/protein_embeddings",
    "cpt":"/home/fangchh/workdir/triplet/data/ESM_CPT/protein_embeddings",
    "sft":"",
    "enz":""
}

DATA_PATHS = {
    "csv_path": "/home/fangchh/workdir/triplet/data/split100.csv",
    "train_csv_path":"train_split_70.csv",
    "val_csv_path":"val_split_70.csv",
    "embedding_dir": embedding_paths["cpt"],
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
    # ============ 用于 final_training 模式 ============
    # 这些值在 eval_hyperparam_search 阶段确定后手动填入
    "default_k": 3,
    "default_tau": 0.35,
}

TUNE_CONFIG = {
    # ============ 用于 eval_hyperparam_search 模式 ============
    'k_search': [3, 5, 7, 10],  # 评估超参数搜索空间
    'tau_search_range': (0.20, 0.50, 10)  # (start, end, num_points)
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
    'temperature': [0.1],
    'patience': [5, 10],
    'P': [16],
    'K': [4],
    'dropout_rate': [0.3, 0.4,0.1,0.2],
}