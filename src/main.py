import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import random
import numpy as np
# 导入 train_test_split 用于动态划分
from sklearn.model_selection import train_test_split
# 导入学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR

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

def set_seeds(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # ==========================================================
    # --- 1. 实验环境设置 ---
    # ==========================================================
    
    # 获取实验模式
    experiment_mode = cfg.EXPERIMENT_MODE
    
    # 根据实验模式生成 run_name
    run_name = cfg.ARTIFACTS_CONFIG.get('run_name')
    if run_name is None:
        mode_suffix = "hp_search" if experiment_mode == "hyperparameter_search" else "final"
        run_name = f"train_{mode_suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # 创建本次运行的专属输出文件夹
    run_output_dir = os.path.join(cfg.ARTIFACTS_CONFIG['output_dir'], run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # 设置日志记录
    log_path = os.path.join(run_output_dir, "experiment.log")
    utils.setup_logging(log_path)
    
    # 使用一个标准的随机种子
    TARGET_SEED = cfg.SEED
    set_seeds(TARGET_SEED)
    # ==========================================================

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
    
    # 2.1 加载所有标签信息和所有 ID
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    
    # 用外部测试集数据更新主映射表
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master.update(id_to_ecs_ext)
    
    # ==========================================================
    # 2.2 根据实验模式决定是否划分训练集和验证集
    # ==========================================================
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    if experiment_mode == "hyperparameter_search":
        # ========== 超参数搜索模式: 使用 train/val 划分 ==========
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
        # ========== 最终训练模式: 使用全部数据训练 ==========
        logging.info("="*60)
        logging.info("MODE: Final Training - Using ALL data for training")
        logging.info("="*60)
        
        train_ids_orig = all_original_ids
        val_ids_orig = []
        
        logging.info(f"Using all {len(train_ids_orig)} samples for training (no validation split)")
        
        # 从配置中读取预先确定的超参数
        default_k = cfg.EVAL_CONFIG.get('default_k')
        default_tau = cfg.EVAL_CONFIG.get('default_tau')
        logging.info(f"Will use pre-determined hyperparameters: k={default_k}, tau={default_tau}")
        
    else:
        raise ValueError(f"Unknown experiment mode: {experiment_mode}. Must be 'hyperparameter_search' or 'final_training'")
    
    # ==========================================================
    # 2.3 使用突变体增强训练数据
    # ==========================================================
    id_to_ecs_master = data_utils.augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=cfg.DATA_PATHS["embedding_dir"],
        train_ids_orig_only=train_ids_orig
    )
    
    # 2.4 构建最终的ID池并过滤掉没有嵌入的样本
    mutants_train = [sid for sid in id_to_ecs_master.keys() if '_' in sid and sid.split('_')[0] in set(train_ids_orig)]
    train_pool_ids_raw = train_ids_orig + mutants_train
    
    train_ids = data_utils.filter_ids_with_embeddings(train_pool_ids_raw, cfg.DATA_PATHS["embedding_dir"], "[TrainPool] ")
    val_ids = data_utils.filter_ids_with_embeddings(val_ids_orig, cfg.DATA_PATHS["embedding_dir"], "[Validation] ")
    
    # 我们的参考库(Gallery)只使用原始的、非突变体的训练样本
    gallery_ids = data_utils.filter_ids_with_embeddings(train_ids_orig, cfg.DATA_PATHS["embedding_dir"], "[Gallery] ")

    # ==========================================================
    # --- 3. 创建 Datasets 和 DataLoaders ---
    # ==========================================================
    logging.info("Creating Datasets and DataLoaders...")

    # 3.1 创建 Datasets
    train_dataset = ContrastiveDataset(train_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    gallery_dataset = EvaluationDataset(gallery_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    
    # 只在超参数搜索模式下创建验证集
    if experiment_mode == "hyperparameter_search":
        val_dataset_for_loss = ContrastiveDataset(val_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
        val_dataset_for_metrics = EvaluationDataset(val_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    else:
        # 最终训练模式下创建空的验证集占位符
        val_dataset_for_loss = None
        val_dataset_for_metrics = None

    # 3.2 创建 DataLoaders
    pk_sampler = PKSampler(
        ec_to_indices=train_dataset.ec_to_indices,
        p=cfg.TRAIN_CONFIG['P'],
        k=cfg.TRAIN_CONFIG['K']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        collate_fn=contrastive_collate_fn,
        **cfg.DATALOADER_CONFIG
    )
    
    batch_size = cfg.TRAIN_CONFIG['P'] * cfg.TRAIN_CONFIG['K']
    
    # 根据模式创建验证集 DataLoader
    if experiment_mode == "hyperparameter_search":
        val_loader_loss = DataLoader(val_dataset_for_loss, batch_size=batch_size, shuffle=False, 
                                      collate_fn=contrastive_collate_fn, **cfg.DATALOADER_CONFIG)
        val_loader_for_metrics = DataLoader(val_dataset_for_metrics, batch_size=batch_size, shuffle=False, 
                                            collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)
    else:
        # 创建空的 DataLoader 占位符
        val_loader_loss = None
        val_loader_for_metrics = None
    
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)

    # --- 创建OOD测试集的 Loaders ---
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
    
    # 初始化学习率调度器
    try:
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cfg.TRAIN_CONFIG['epochs'],
            eta_min=cfg.TRAIN_CONFIG.get('scheduler_eta_min', 1e-6)
        )
        logging.info(f"Using CosineAnnealingLR scheduler with T_max={cfg.TRAIN_CONFIG['epochs']}.")
    except KeyError:
        logging.error("错误: `cfg.TRAIN_CONFIG['epochs']` 未设置。")
        logging.warning("将不使用学习率调度器。")
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
            "EXPERIMENT_MODE": experiment_mode  # 传递实验模式
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