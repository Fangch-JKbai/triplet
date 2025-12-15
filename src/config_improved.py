# config_improved.py
import torch
from datetime import datetime
import numpy as np

SEED = 42
N_SPLITS = 5
DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 新增：实验模式配置
# ============================================================================
COMPARISON_MODE = "embedding_only"  # "comprehensive" | "embedding_only" | "full_pipeline"

ARTIFACTS_CONFIG = {
    "output_dir": "experiments",
    "run_name": 'base_1117'
}

# ============================================================================
# 模型配置 - 根据模式动态调整
# ============================================================================
if COMPARISON_MODE == "embedding_only":
    # 只用线性层，最小化下游任务影响
    MODEL_CONFIG = {
        'input_dim': 1280,
        'd_z': 128,
        'architecture': 'linear',
        'dropout_rate': 0.2,
    }
elif COMPARISON_MODE == "full_pipeline":
    # 完整MLP，测试端到端性能
    MODEL_CONFIG = {
        'input_dim': 1280,
        'd1': 512,
        'd2': 256,
        'd_z': 128,
        'architecture': 'mlp',
        'dropout_rate': 0.2,
    }
else:  # comprehensive - 两种都测试
    MODEL_CONFIG = {
        'input_dim': 1280,
        'd1': 512,
        'd2': 256,
        'd_z': 128,
        'architecture': 'mlp',  # 主要用MLP
        'dropout_rate': 0.2,
    }

TRAIN_CONFIG = {
    "P": 16,
    "K": 4,
    "temperature": 0.1,
    "epochs":300,
    "learning_rate": 1e-4,
    "grad_clip_norm": 1.0,
    "patience": 20
}

DATA_PATHS = {
    "csv_path": "/home/fangchh/workdir/triplet/data/split70.csv",
    "train_csv_path": "train_split_70.csv",
    "val_csv_path": "val_split_70.csv",
    "embedding_dir": "/home/fangchh/workdir/triplet/data/base/protein_embeddings",
    "external_test_sets": {
        "price": "/home/fangchh/workdir/triplet/data/price.csv",
        "new": "/home/fangchh/workdir/triplet/data/new.csv"
    }
}

# ============================================================================
# 评估配置 - 扩展超参数搜索空间
# ============================================================================
EVAL_CONFIG = {
    "weighted": True, 
}

if COMPARISON_MODE == "embedding_only":
    # 更大的搜索空间，让不同embedding找到各自最优点
    TUNE_CONFIG = {
        'k_search': [3],
        'tau_search_range': (0.35, 0.35, 1)  # 从0.1到0.7，13个点
    }
else:
    # 常规搜索
    TUNE_CONFIG = {
        'k_search': [3],
        'tau_search_range': (0.35, 0.35, 1)
    }

DATALOADER_CONFIG = {
    "num_workers": 16,
    "pin_memory": True,
    "persistent_workers": True if torch.cuda.is_available() else False, 
    "prefetch_factor": 2,
}

# ============================================================================
# 新增：Embedding质量内在评估配置
# ============================================================================
INTRINSIC_EVAL_CONFIG = {
    "enable": True,  # 是否启用embedding内在质量评估
    
    "clustering_metrics": True,  # 计算聚类指标 (Silhouette, Davies-Bouldin)
    
    "knn_evaluation": {
        "enable": True,
        "k_values": [1, 3, 5, 10],  # 测试不同k值的kNN准确率
    },
    
    "ec_frequency_analysis": {
        "enable": True,
        "thresholds": {
            "common": 100,   # >100 samples
            "medium": 10,    # 10-100 samples
            "rare": 0        # <10 samples
        }
    },
    
    "similarity_distribution": {
        "enable": True,
        "num_bins": 50,  # 相似度分布直方图的bins数量
    }
}

# ============================================================================
# 新增：可视化配置
# ============================================================================
VISUALIZATION_CONFIG = {
    "enable": True,
    
    "tsne": {
        "enable": True,
        "perplexity": 30,
        "n_iter": 1000,
        "max_samples": 2000,  # 最多采样多少个点做t-SNE (太多会很慢)
    },
    
    "confusion_matrix": {
        "enable": True,
        "top_k_ecs": 20,  # 只展示最常见的K个EC
    },
    
    "performance_heatmap": {
        "enable": True,  # k vs tau的性能热图
    }
}