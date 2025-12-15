"""
配置文件模板 - 复制这个文件到 src/config.py 并根据你的需求修改

使用方法：
1. 复制: cp config_template.py src/config.py
2. 修改下面标记为 [MUST CHANGE] 的配置
3. 根据需要调整其他参数
"""

import torch
from datetime import datetime
import numpy as np

# =============================================================================
# 基础配置
# =============================================================================
SEED = 42
N_SPLITS = 5  # K折交叉验证的折数
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # [修改] GPU设备号

# =============================================================================
# 实验模式配置
# =============================================================================
# [MUST CHANGE] 根据你当前的训练阶段选择：
# - "eval_hyperparam_search": 带验证集的训练（用于五折交叉验证或超参数搜索）
# - "final_training": 使用全部数据的最终训练
EXPERIMENT_MODE = "eval_hyperparam_search"  # 第一步：五折交叉验证时用这个
# EXPERIMENT_MODE = "final_training"        # 第二步：最终训练时改成这个

# =============================================================================
# 输出配置
# =============================================================================
ARTIFACTS_CONFIG = {
    "output_dir": "experiments",
    "run_name": None  # None表示自动生成时间戳名称
}

# =============================================================================
# 模型配置
# =============================================================================
MODEL_CONFIG = {
    'input_dim': 1280,      # ESM embedding维度（通常是1280）
    'd1': 512,              # 第一层隐藏层维度（当前代码未使用）
    'd2': 256,              # 第二层隐藏层维度（当前代码未使用）
    'd_z': 128,             # 最终嵌入维度
    'dropout_rate': 0.2,    # Dropout概率
}

# =============================================================================
# 训练配置
# =============================================================================
TRAIN_CONFIG = {
    # PK Sampling参数
    "P": 16,              # batch中的类别数（EC号数量）
    "K": 4,               # 每个类别的样本数
                          # 实际batch_size = P * K = 64

    # 对比学习参数
    "temperature": 0.1,   # SupCon loss的温度参数

    # 训练轮数
    "epochs": 1000,       # 最大训练轮数
                          # 注意：
                          # - eval_hyperparam_search模式下会早停
                          # - final_training模式下训练固定轮数（改成第一步得到的平均epoch）

    # 优化器参数
    "learning_rate": 1e-3,  # 学习率

    # 梯度裁剪
    "grad_clip_norm": 0,    # 0表示不使用梯度裁剪

    # 早停参数
    "patience": 15,         # 早停patience（验证损失多少轮不下降就停止）

    # 学习率调度
    "scheduler_eta_min": 1e-6,  # CosineAnnealing最小学习率
}

# =============================================================================
# 数据路径配置
# =============================================================================
# [MUST CHANGE] 修改为你的实际数据路径！

# 可选：如果你有多个embedding版本，可以在这里配置
embedding_paths = {
    "esm": "/path/to/base/protein_embeddings",
    "cpt": "/path/to/ESM_CPT/protein_embeddings",
    "sft": "/path/to/ESM_SFT/protein_embeddings",
    "enz": "/path/to/ESM_ENZ/protein_embeddings"
}

DATA_PATHS = {
    # [MUST CHANGE] 主训练数据CSV文件
    "csv_path": "/home/fangchh/workdir/triplet/data/split100.csv",

    # [可选] 如果使用固定的train/val划分（当前代码未使用这些）
    "train_csv_path": "train_split_70.csv",
    "val_csv_path": "val_split_70.csv",

    # [MUST CHANGE] Embedding文件目录
    "embedding_dir": embedding_paths["cpt"],  # 选择使用哪个embedding版本

    # [MUST CHANGE] 外部测试集
    "external_test_sets": {
        "price": "/home/fangchh/workdir/triplet/data/price.csv",
        "new": "/home/fangchh/workdir/triplet/data/new.csv"
    }
}

# =============================================================================
# 数据划分配置（用于动态划分train/val）
# =============================================================================
DATA_CONFIG = {
    'val_split_ratio': 0.2,   # 验证集比例（仅在eval_hyperparam_search模式使用）
    'stratify_level': 3,      # 分层采样的EC号级别（1-4）
}

# =============================================================================
# 评估配置
# =============================================================================
EVAL_CONFIG = {
    "weighted": True,  # 是否使用加权评估（根据类别频率加权）

    # [重要] final_training模式下使用的默认超参数
    # 这些值应该在第一步（五折交叉验证）完成后，根据验证集结果填入
    "default_k": 3,      # KNN的K值
    "default_tau": 0.35, # 投票阈值
}

# =============================================================================
# 超参数调优配置（用于在验证集上搜索最佳k和tau）
# =============================================================================
TUNE_CONFIG = {
    'k_search': [3, 5, 7, 10],           # 搜索的K值列表
    'tau_search_range': (0.20, 0.50, 10) # (起始, 结束, 点数)
}

# =============================================================================
# DataLoader配置
# =============================================================================
DATALOADER_CONFIG = {
    "num_workers": 16,       # 数据加载的进程数（根据CPU核心数调整）
    "pin_memory": True,      # 固定内存（GPU训练时建议True）
    "persistent_workers": True if torch.cuda.is_available() else False,
    "prefetch_factor": 2,    # 预取batch数量
}

# =============================================================================
# 超参数搜索空间（用于train_hyperparam_search模式，当前代码可能未使用）
# =============================================================================
HYPERPARAM_SEARCH_SPACE = {
    'learning_rate': [1e-3, 1e-4],
    'temperature': [0.1],
    'patience': [5, 10],
    'P': [16],
    'K': [4],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
}

# =============================================================================
# 配置验证（可选）
# =============================================================================
def validate_config():
    """验证配置的有效性"""
    import os

    errors = []
    warnings = []

    # 检查必要的数据路径
    if not os.path.exists(DATA_PATHS["csv_path"]):
        errors.append(f"主数据集不存在: {DATA_PATHS['csv_path']}")

    if not os.path.exists(DATA_PATHS["embedding_dir"]):
        errors.append(f"Embedding目录不存在: {DATA_PATHS['embedding_dir']}")

    for name, path in DATA_PATHS["external_test_sets"].items():
        if not os.path.exists(path):
            warnings.append(f"测试集 {name} 不存在: {path}")

    # 检查实验模式
    if EXPERIMENT_MODE not in ["eval_hyperparam_search", "final_training"]:
        errors.append(f"未知的实验模式: {EXPERIMENT_MODE}")

    # 检查final_training模式的配置
    if EXPERIMENT_MODE == "final_training":
        if TRAIN_CONFIG["epochs"] > 100:
            warnings.append(
                f"final_training模式下epochs={TRAIN_CONFIG['epochs']}过大，"
                "建议使用五折交叉验证得到的平均epoch数"
            )

    # 打印结果
    if errors:
        print("配置错误:")
        for err in errors:
            print(f"  ✗ {err}")
        return False

    if warnings:
        print("配置警告:")
        for warn in warnings:
            print(f"  ⚠ {warn}")

    return True

# 可选：导入时自动验证
# validate_config()
