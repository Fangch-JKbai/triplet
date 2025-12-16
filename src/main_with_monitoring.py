import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

# 导入项目模块
import config as cfg
import data_utils
import utils
from model import EcClassifier
from datasets import ContrastiveDataset, EvaluationDataset, contrastive_collate_fn
from evaluation import evaluation_collate_fn
from loss import SupervisedContrastiveLoss
from samplers import PKSampler
from trainer_with_monitoring import ExperimentRunnerWithMonitoring  # 使用新的trainer

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

    experiment_mode = cfg.EXPERIMENT_MODE

    # 生成 run_name
    run_name = cfg.ARTIFACTS_CONFIG.get('run_name')
    if run_name is None:
        mode_suffix = "monitored"
        run_name = f"train_{mode_suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # 创建输出文件夹
    run_output_dir = os.path.join(cfg.ARTIFACTS_CONFIG['output_dir'], run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    # 设置日志
    log_path = os.path.join(run_output_dir, "experiment.log")
    utils.setup_logging(log_path)

    # 设置随机种子
    TARGET_SEED = cfg.SEED
    set_seeds(TARGET_SEED)

    logging.info("="*60)
    logging.info(f"EXPERIMENT MODE: {experiment_mode.upper()} (WITH MONITORING)")
    logging.info("="*60)
    logging.info(f"Experiment run: {run_name}")
    logging.info(f"Artifacts will be saved to: {run_output_dir}")
    logging.info(f"Using device: {cfg.DEVICE}")
    logging.info(f"Using fixed random seed: {TARGET_SEED}")

    # ==========================================================
    # --- 2. 数据准备 ---
    # ==========================================================
    logging.info("Preparing data...")

    # 加载标签映射
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])

    # 加载外部测试集
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master.update(id_to_ecs_ext)

    # 获取原始ID
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])

    # ==========================================================
    # 数据划分（保留验证集用于early stopping）
    # ==========================================================
    logging.info("="*60)
    logging.info("MODE: Training with validation set for early stopping")
    logging.info("="*60)

    val_ratio = 0.2  # 20%验证集
    logging.info(f"Using validation split ratio: {val_ratio}")

    # 尝试分层采样
    try:
        stratify_level = cfg.DATA_CONFIG.get('stratify_level', 3)
        labels_for_stratify = [
            '.'.join(id_to_ecs_master[uid][0].split('.')[:stratify_level])
            for uid in all_original_ids
        ]

        train_ids_orig, val_ids_orig = train_test_split(
            all_original_ids,
            test_size=val_ratio,
            random_state=TARGET_SEED,
            stratify=labels_for_stratify
        )
        logging.info(f"Successfully used stratified split (level {stratify_level})")

    except ValueError as e:
        logging.warning(f"Stratified split failed: {e}")
        logging.warning("Falling back to random split")

        train_ids_orig, val_ids_orig = train_test_split(
            all_original_ids,
            test_size=val_ratio,
            random_state=TARGET_SEED,
            stratify=None
        )

    logging.info(f"Split {len(all_original_ids)} original IDs into:")
    logging.info(f"  - Train: {len(train_ids_orig)} samples")
    logging.info(f"  - Val:   {len(val_ids_orig)} samples")

    # 数据增强
    id_to_ecs_master = data_utils.augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=cfg.DATA_PATHS["embedding_dir"],
        train_ids_orig_only=train_ids_orig
    )

    # 构建ID池
    mutants_train = [sid for sid in id_to_ecs_master.keys()
                     if '_' in sid and sid.split('_')[0] in set(train_ids_orig)]
    train_pool_ids_raw = train_ids_orig + mutants_train

    # 过滤ID
    train_ids = data_utils.filter_ids_with_embeddings(
        train_pool_ids_raw, cfg.DATA_PATHS["embedding_dir"], "[TrainPool] ")
    val_ids = data_utils.filter_ids_with_embeddings(
        val_ids_orig, cfg.DATA_PATHS["embedding_dir"], "[Validation] ")
    gallery_ids = data_utils.filter_ids_with_embeddings(
        train_ids_orig, cfg.DATA_PATHS["embedding_dir"], "[Gallery] ")

    # ==========================================================
    # --- 3. 创建 Datasets 和 DataLoaders ---
    # ==========================================================
    logging.info("Creating Datasets and DataLoaders...")

    train_dataset = ContrastiveDataset(train_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    val_dataset_for_loss = ContrastiveDataset(val_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    val_dataset_for_metrics = EvaluationDataset(val_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    gallery_dataset = EvaluationDataset(gallery_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])

    # 创建DataLoaders
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

    # 验证集DataLoader（用更大的batch size）
    val_batch_size = 256
    val_loader_loss = DataLoader(
        val_dataset_for_loss,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=contrastive_collate_fn,
        num_workers=cfg.DATALOADER_CONFIG['num_workers'],
        pin_memory=cfg.DATALOADER_CONFIG['pin_memory']
    )
    val_loader_for_metrics = DataLoader(
        val_dataset_for_metrics,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=evaluation_collate_fn,
        **cfg.DATALOADER_CONFIG
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=evaluation_collate_fn,
        **cfg.DATALOADER_CONFIG
    )

    logging.info(f"Train loader: {len(train_loader)} batches")
    logging.info(f"Val loader (loss): {len(val_loader_loss)} batches (batch_size={val_batch_size})")
    logging.info(f"Gallery loader: {len(gallery_loader)} batches")

    # 创建测试集Loaders
    ood_test_loaders = {}
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        _, ext_test_ids = data_utils.load_and_create_label_map(path)
        ext_test_dataset = EvaluationDataset(
            ext_test_ids, id_to_ecs_master,
            cfg.DATA_PATHS["embedding_dir"],
            strict_embeddings=False
        )
        ood_test_loaders[name] = DataLoader(
            ext_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=evaluation_collate_fn,
            **cfg.DATALOADER_CONFIG
        )
        logging.info(f"Test loader ({name}): {len(ood_test_loaders[name])} batches")

    # ==========================================================
    # --- 4. 初始化模型、损失函数、优化器和调度器 ---
    # ==========================================================
    logging.info("Initializing model, criterion, optimizer, and scheduler...")
    model = EcClassifier(**cfg.MODEL_CONFIG).to(cfg.DEVICE)
    criterion = SupervisedContrastiveLoss(temperature=cfg.TRAIN_CONFIG['temperature'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN_CONFIG['learning_rate'])

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.TRAIN_CONFIG['epochs'],
        eta_min=cfg.TRAIN_CONFIG.get('scheduler_eta_min', 1e-6)
    )

    # ==========================================================
    # --- 5. 启动实验（使用监控版trainer） ---
    # ==========================================================

    # 每隔多少epoch评估测试集（可以设置为1, 5, 10等）
    EVAL_INTERVAL = 5  # 每5个epoch评估一次，可以改成1

    runner = ExperimentRunnerWithMonitoring(
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
        run_output_dir=run_output_dir,
        eval_interval=EVAL_INTERVAL  # 评估间隔
    )

    logging.info(f"Will evaluate test sets every {EVAL_INTERVAL} epoch(s)")

    runner.run(
        train_loader=train_loader,
        val_loader_loss=val_loader_loss,
        val_loader_metrics=val_loader_for_metrics,
        gallery_loader=gallery_loader,
        ood_test_loaders=ood_test_loaders
    )

    logging.info(f"Experiment {run_name} finished successfully.")
    logging.info(f"Results saved in: {run_output_dir}")
    logging.info(f"Performance curves: {run_output_dir}/test_performance_curve.png")

if __name__ == '__main__':
    main()
