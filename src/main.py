# main.py (Updated with Augmented Gallery + train_one_run API)
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import random
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Any

# 导入我们项目的所有模块
import config as cfg
import data_utils
import utils
from model import EcClassifier
from datasets import ContrastiveDataset, EvaluationDataset, contrastive_collate_fn
from evaluation import evaluation_collate_fn
from loss import SupervisedContrastiveLoss
from samplers import PKSampler
from trainer import ExperimentRunner
from schedulers import create_scheduler


def set_seeds(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_run(
    embedding_dir: str,
    P: int,
    K: int,
    output_dir: str,
    seed: int = 42,
    epochs_override: Optional[int] = None,
    device: Optional[torch.device] = None,
    csv_path: Optional[str] = None,
    external_test_sets: Optional[Dict[str, str]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    train_config_override: Optional[Dict[str, Any]] = None,
) -> str:
    """
    独立的训练入口函数，用于调参脚本调用。
    
    Args:
        embedding_dir: embedding 文件目录
        P: PKSampler 的 P 参数
        K: PKSampler 的 K 参数
        output_dir: 输出目录
        seed: 随机种子
        epochs_override: 覆盖 epochs 数量（用于调参时减少训练时间）
        device: 训练设备
        csv_path: 训练数据 CSV 路径
        external_test_sets: 外部测试集路径字典（调参时通常为空）
        model_config: 模型配置覆盖
        train_config_override: 训练配置覆盖
    
    Returns:
        best_model_path: 最佳模型的路径
    """
    # 设置默认值
    if device is None:
        device = cfg.DEVICE
    if csv_path is None:
        csv_path = cfg.DATA_PATHS["csv_path"]
    if external_test_sets is None:
        external_test_sets = {}  # 调参时不需要测试集
    if model_config is None:
        model_config = cfg.MODEL_CONFIG.copy()
    
    # 合并训练配置
    train_config = cfg.TRAIN_CONFIG.copy()
    if train_config_override:
        train_config.update(train_config_override)
    
    # 覆盖 P, K, epochs
    train_config['P'] = P
    train_config['K'] = K
    if epochs_override is not None:
        train_config['epochs'] = epochs_override
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(output_dir, "train.log")
    utils.setup_logging(log_path)
    
    # 设置随机种子
    set_seeds(seed)
    
    logging.info("="*60)
    logging.info(f"train_one_run: P={P}, K={K}, epochs={train_config['epochs']}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Embedding dir: {embedding_dir}")
    logging.info("="*60)
    
    # ==========================================================
    # --- 数据准备 ---
    # ==========================================================
    logging.info("Preparing data...")
    
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(csv_path)
    
    # 加载外部测试集标签映射
    for name, path in external_test_sets.items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master.update(id_to_ecs_ext)
    
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    # 使用全量数据训练（final_training 模式）
    train_ids_orig = all_original_ids
    
    logging.info(f"Using all {len(train_ids_orig)} samples for training")
    
    # 使用突变体增强训练数据
    id_to_ecs_master = data_utils.augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=embedding_dir,
        train_ids_orig_only=train_ids_orig
    )
    
    mutants_train = [sid for sid in id_to_ecs_master.keys() 
                    if '_' in sid and sid.split('_')[0] in set(train_ids_orig)]
    
    train_pool_ids_raw = train_ids_orig + mutants_train
    
    # 检查 Embeddings 文件是否存在
    train_ids = data_utils.filter_ids_with_embeddings(
        train_pool_ids_raw, embedding_dir, "[TrainPool] "
    )
    gallery_ids = data_utils.filter_ids_with_embeddings(
        train_pool_ids_raw, embedding_dir, "[Gallery-Augmented] "
    )
    
    # ==========================================================
    # --- 创建 Datasets 和 DataLoaders ---
    # ==========================================================
    logging.info("Creating Datasets and DataLoaders...")
    
    train_dataset = ContrastiveDataset(train_ids, id_to_ecs_master, embedding_dir)
    gallery_dataset = EvaluationDataset(gallery_ids, id_to_ecs_master, embedding_dir)
    
    pk_sampler = PKSampler(
        ec_to_indices=train_dataset.ec_to_indices,
        p=P,
        k=K,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        collate_fn=contrastive_collate_fn,
        **cfg.DATALOADER_CONFIG
    )
    
    batch_size = P * K
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG
    )
    
    # OOD 测试集（调参时通常为空）
    ood_test_loaders = {}
    for name, path in external_test_sets.items():
        _, ext_test_ids = data_utils.load_and_create_label_map(path)
        ext_test_dataset = EvaluationDataset(
            ext_test_ids, id_to_ecs_master, embedding_dir, strict_embeddings=False
        )
        ood_test_loaders[name] = DataLoader(
            ext_test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG
        )
    
    # ==========================================================
    # --- 初始化模型、损失函数、优化器和调度器 ---
    # ==========================================================
    logging.info("Initializing model, criterion, optimizer, and scheduler...")
    model = EcClassifier(**model_config).to(device)
    criterion = SupervisedContrastiveLoss(temperature=train_config['temperature'])
    optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate'])
    
    scheduler_type = train_config.get('scheduler_type', 'cosine_warmup')
    warmup_epochs = train_config.get('warmup_epochs', None)
    warmup_ratio = train_config.get('warmup_ratio', 0.1)
    min_lr = train_config.get('scheduler_min_lr', 1e-6)
    
    try:
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_epochs=train_config['epochs'],
            warmup_epochs=warmup_epochs,
            warmup_ratio=warmup_ratio,
            min_lr=min_lr
        )
    except Exception as e:
        logging.error(f"Failed to create scheduler: {e}")
        scheduler = None
    
    # ==========================================================
    # --- 启动训练 ---
    # ==========================================================
    # 构建 runner 需要的 config
    runner_config = {
        "TRAIN_CONFIG": train_config,
        "EVAL_CONFIG": cfg.EVAL_CONFIG,
        "TUNE_CONFIG": cfg.TUNE_CONFIG,
        "EXPERIMENT_MODE": "final_training"  # 调参时也用 final_training 模式（无验证集）
    }
    
    runner = ExperimentRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=runner_config,
        device=device,
        run_output_dir=output_dir
    )
    
    # 运行训练（调参时不需要 OOD 测试）
    runner.run(
        train_loader=train_loader,
        val_loader_loss=None,
        val_loader_metrics=None,
        gallery_loader=gallery_loader,
        ood_test_loaders=ood_test_loaders
    )
    
    best_model_path = os.path.join(output_dir, "best_model.pth")
    logging.info(f"Training finished. Best model saved to: {best_model_path}")
    
    return best_model_path


def main():
    """原有的主函数，保持不变"""
    # ==========================================================
    # --- 1. 实验环境设置 ---
    # ==========================================================
    
    experiment_mode = cfg.EXPERIMENT_MODE
    
    run_name = cfg.ARTIFACTS_CONFIG.get('run_name')
    if run_name is None:
        mode_suffix = "hp_search" if experiment_mode == "hyperparameter_search" else "final"
        run_name = f"train_{mode_suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    run_output_dir = os.path.join(cfg.ARTIFACTS_CONFIG['output_dir'], run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    log_path = os.path.join(run_output_dir, "experiment.log")
    utils.setup_logging(log_path)
    
    TARGET_SEED = cfg.SEED
    set_seeds(TARGET_SEED)

    logging.info("="*60)
    logging.info(f"EXPERIMENT MODE: {experiment_mode.upper()}")
    logging.info("="*60)
    logging.info(f"Experiment run: {run_name}")
    logging.info(f"Artifacts will be saved to: {run_output_dir}")
    logging.info(f"Using device: {cfg.DEVICE}")
    logging.info(f"Using fixed random seed: {TARGET_SEED}")

    # ==========================================================
    # --- 2. 数据准备 (Data Preparation) ---
    # ==========================================================
    logging.info("Preparing data...")
    
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    
    # 加载外部测试集标签映射（确保评估时标签字典完整）
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master.update(id_to_ecs_ext)
    
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    if experiment_mode == "hyperparameter_search":
        logging.info("="*60)
        logging.info("MODE: Hyperparameter Search - Splitting data into train/val")
        logging.info("="*60)
        
        try:
            stratify_level = cfg.DATA_CONFIG.get('stratify_level', 3) 
            labels_for_stratify = [
                '.'.join(id_to_ecs_master[uid][0].split('.')[:stratify_level]) 
                for uid in all_original_ids
            ]
        except Exception as e:
            logging.warning(f"无法创建分层标签 (级别 {stratify_level}): {e}。将不使用分层抽样。")
            labels_for_stratify = None

        val_ratio = cfg.DATA_CONFIG.get('val_split_ratio', 0.2)
        logging.info(f"Using validation split ratio: {val_ratio}")

        train_ids_orig, val_ids_orig = train_test_split(
            all_original_ids,
            test_size=val_ratio,
            random_state=TARGET_SEED,
            stratify=labels_for_stratify
        )

        logging.info(f"Split {len(all_original_ids)} original IDs into:")
        logging.info(f"  - Train: {len(train_ids_orig)} samples")
        logging.info(f"  - Val:   {len(val_ids_orig)} samples")
        
    elif experiment_mode == "final_training":
        logging.info("="*60)
        logging.info("MODE: Final Training - Using ALL data for training")
        logging.info("="*60)
        
        train_ids_orig = all_original_ids
        val_ids_orig = []
        
        logging.info(f"Using all {len(train_ids_orig)} samples for training (no validation split)")
        
        default_k = cfg.EVAL_CONFIG.get('default_k')
        default_tau = cfg.EVAL_CONFIG.get('default_tau')
        logging.info(f"Will use pre-determined hyperparameters: k={default_k}, tau={default_tau}")
        
    else:
        raise ValueError(f"Unknown experiment mode: {experiment_mode}. Must be 'hyperparameter_search' or 'final_training'")
    
    # ==========================================================
    # 2.3 使用突变体增强训练数据 & 检索库
    # ==========================================================
    id_to_ecs_master = data_utils.augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=cfg.DATA_PATHS["embedding_dir"],
        train_ids_orig_only=train_ids_orig
    )
    
    # 筛选出属于训练集的突变体
    mutants_train = [sid for sid in id_to_ecs_master.keys() if '_' in sid and sid.split('_')[0] in set(train_ids_orig)]
    
    # 组合成训练池 (Originals + Mutants)
    train_pool_ids_raw = train_ids_orig + mutants_train
    
    # 检查 Embeddings 文件是否存在
    train_ids = data_utils.filter_ids_with_embeddings(train_pool_ids_raw, cfg.DATA_PATHS["embedding_dir"], "[TrainPool] ")
    val_ids = data_utils.filter_ids_with_embeddings(val_ids_orig, cfg.DATA_PATHS["embedding_dir"], "[Validation] ")
    
    # === 修改点：使用增强的 Gallery (Originals + Mutants) ===
    logging.info("Using AUGMENTED Gallery (Originals + Mutants) for retrieval support...")
    gallery_ids = data_utils.filter_ids_with_embeddings(train_pool_ids_raw, cfg.DATA_PATHS["embedding_dir"], "[Gallery-Augmented] ")

    # ==========================================================
    # --- 3. 创建 Datasets 和 DataLoaders ---
    # ==========================================================
    logging.info("Creating Datasets and DataLoaders...")

    train_dataset = ContrastiveDataset(train_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    gallery_dataset = EvaluationDataset(gallery_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    
    if experiment_mode == "hyperparameter_search":
        val_dataset_for_loss = ContrastiveDataset(val_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
        val_dataset_for_metrics = EvaluationDataset(val_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    else:
        val_dataset_for_loss = None
        val_dataset_for_metrics = None

    pk_sampler = PKSampler(
        ec_to_indices=train_dataset.ec_to_indices,
        p=cfg.TRAIN_CONFIG['P'],
        k=cfg.TRAIN_CONFIG['K'],
        seed=TARGET_SEED,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        collate_fn=contrastive_collate_fn,
        **cfg.DATALOADER_CONFIG
    )
    
    batch_size = cfg.TRAIN_CONFIG['P'] * cfg.TRAIN_CONFIG['K']
    
    if experiment_mode == "hyperparameter_search":
        val_loader_loss = DataLoader(val_dataset_for_loss, batch_size=batch_size, shuffle=False, 
                                      collate_fn=contrastive_collate_fn, **cfg.DATALOADER_CONFIG)
        val_loader_for_metrics = DataLoader(val_dataset_for_metrics, batch_size=batch_size, shuffle=False, 
                                            collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)
    else:
        val_loader_loss = None
        val_loader_for_metrics = None
    
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)

    ood_test_loaders = {}
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        _, ext_test_ids = data_utils.load_and_create_label_map(path)
        ext_test_dataset = EvaluationDataset(ext_test_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"], strict_embeddings=False)
        ood_test_loaders[name] = DataLoader(ext_test_dataset, batch_size=batch_size, shuffle=False, 
                                           collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)
    
    # ==========================================================
    # --- 4. 初始化模型、损失函数、优化器和调度器 ---
    # ==========================================================
    logging.info("Initializing model, criterion, optimizer, and scheduler...")
    model = EcClassifier(**cfg.MODEL_CONFIG).to(cfg.DEVICE)
    criterion = SupervisedContrastiveLoss(temperature=cfg.TRAIN_CONFIG['temperature'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN_CONFIG['learning_rate'])
    
    scheduler_type = cfg.TRAIN_CONFIG.get('scheduler_type', 'cosine_warmup')
    warmup_epochs = cfg.TRAIN_CONFIG.get('warmup_epochs', None)
    warmup_ratio = cfg.TRAIN_CONFIG.get('warmup_ratio', 0.1)
    min_lr = cfg.TRAIN_CONFIG.get('scheduler_min_lr', 1e-6)
    
    try:
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_epochs=cfg.TRAIN_CONFIG['epochs'],
            warmup_epochs=warmup_epochs,
            warmup_ratio=warmup_ratio,
            min_lr=min_lr
        )
        
        effective_warmup = warmup_epochs if warmup_epochs else int(cfg.TRAIN_CONFIG['epochs'] * warmup_ratio)
        logging.info(f"Using {scheduler_type} scheduler:")
        logging.info(f"  - Total epochs: {cfg.TRAIN_CONFIG['epochs']}")
        logging.info(f"  - Warmup epochs: {effective_warmup}")
        logging.info(f"  - Min LR: {min_lr}")
        
    except Exception as e:
        logging.error(f"Failed to create scheduler: {e}")
        logging.warning("Falling back to no scheduler.")
        scheduler = None
    
    # ==========================================================
    # --- 5. 启动实验 ---
    # ==========================================================
    runner = ExperimentRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config={ 
            "TRAIN_CONFIG": cfg.TRAIN_CONFIG,
            "EVAL_CONFIG": cfg.EVAL_CONFIG,
            "TUNE_CONFIG": cfg.TUNE_CONFIG,
            "EXPERIMENT_MODE": experiment_mode
        },
        device=cfg.DEVICE,
        run_output_dir=run_output_dir
    )
    
    runner.run(
        train_loader=train_loader,
        val_loader_loss=val_loader_loss,
        val_loader_metrics=val_loader_for_metrics,
        gallery_loader=gallery_loader,
        ood_test_loaders=ood_test_loaders
    )
    logging.info(f"Experiment {run_name} finished successfully.")


if __name__ == '__main__':
    main()