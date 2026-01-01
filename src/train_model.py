# train_final_model.py
# (此脚本从一个固定的 train/val 划分 训练一个最终模型)

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import random
import numpy as np
import json 

# 导入我们项目的所有模块
import config as cfg
import data_utils
import utils
from model import EcClassifier
from datasets import ContrastiveDataset, EvaluationDataset, contrastive_collate_fn
from evaluation import evaluation_collate_fn
from loss import SupervisedContrastiveLoss
from triplet.src.pk_sampler_v1_backup import PKSampler
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

# --- [学术配置] ---
# (这些文件必须由 create_fixed_split.py 生成)
TRAIN_IDS_FILE = "fixed_split_train_ids.json"
VAL_IDS_FILE = "fixed_split_val_ids.json"
# ----------------------

def main():
    # ==========================================================
    # --- 1. 实验环境设置 ---
    # ==========================================================
    
    run_name = cfg.ARTIFACTS_CONFIG.get('run_name')
    if run_name is None:
        run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_Final_Run"
    
    run_output_dir = os.path.join(cfg.ARTIFACTS_CONFIG['output_dir'], run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    utils.setup_logging(os.path.join(run_output_dir, "run.log"))
    
    # 再次设置种子，确保 DataLoader 等操作也是可复现的
    set_seeds(cfg.SEED) 

    logging.info(f"*** 最终模型训练 (单一固定划分) ***")
    logging.info(f"实验名称: {run_name}")
    logging.info(f"实验结果目录: {run_output_dir}")
    logging.info(f"使用设备: {cfg.DEVICE}")

    # ==========================================================
    # --- 2. 准备 OOD (分布外) 测试集 ---
    # ==========================================================
    logging.info("正在准备 OOD (PRICE, NEW) 测试集...")
    
    # 2.1 加载主标签映射
    id_to_ecs_master_base, _ = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master_base.update(id_to_ecs_ext)
    
    # 2.2 创建 OOD Dataloaders
    ood_test_loaders = {}
    eval_batch_size = cfg.TRAIN_CONFIG['P'] * cfg.TRAIN_CONFIG['K']
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        _, ext_test_ids = data_utils.load_and_create_label_map(path)
        ext_test_dataset = EvaluationDataset(ext_test_ids, id_to_ecs_master_base, cfg.DATA_PATHS["embedding_dir"], strict_embeddings=False)
        ood_test_loaders[name] = DataLoader(ext_test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)
    
    logging.info(f"已创建 {len(ood_test_loaders)} 个 OOD 测试集。")

    # ==========================================================
    # --- 3. 加载固定的训练/验证集ID ---
    # ==========================================================
    
    logging.info(f"正在从 {TRAIN_IDS_FILE} 加载固定的训练集ID...")
    try:
        with open(TRAIN_IDS_FILE, 'r') as f:
            train_ids_orig = json.load(f)
    except FileNotFoundError:
        logging.error(f"[错误] 未找到 {TRAIN_IDS_FILE}！请先运行 create_fixed_split.py。")
        return

    logging.info(f"正在从 {VAL_IDS_FILE} 加载固定的验证集ID...")
    try:
        with open(VAL_IDS_FILE, 'r') as f:
            val_ids_orig = json.load(f)
    except FileNotFoundError:
        logging.error(f"[错误] 未找到 {VAL_IDS_FILE}！请先运行 create_fixed_split.py。")
        return

    logging.info(f"固定训练集 (原始ID): {len(train_ids_orig)}")
    logging.info(f"固定验证集 (原始ID): {len(val_ids_orig)}")

    # 3.3 数据增强和过滤 (与 K-Fold 循环 内的逻辑完全一致)
    
    id_to_ecs_fold = data_utils.augment_map_with_mutants(
        id_to_ecs_master_base.copy(), 
        embedding_dir=cfg.DATA_PATHS["embedding_dir"],
        train_ids_orig_only=train_ids_orig 
    )
    
    mutants_train = [sid for sid in id_to_ecs_fold.keys() if '_' in sid and sid.split('_')[0] in set(train_ids_orig)]
    train_pool_ids_raw = train_ids_orig + mutants_train
    
    train_ids = data_utils.filter_ids_with_embeddings(train_pool_ids_raw, cfg.DATA_PATHS["embedding_dir"], "[Final TrainPool] ")
    val_ids = data_utils.filter_ids_with_embeddings(val_ids_orig, cfg.DATA_PATHS["embedding_dir"], "[Final Validation] ")
    gallery_ids = data_utils.filter_ids_with_embeddings(train_ids_orig, cfg.DATA_PATHS["embedding_dir"], "[Final Gallery] ")

    # 3.4 创建 DataLoaders
    logging.info("正在为最终运行创建 Datasets 和 DataLoaders...")

    train_dataset = ContrastiveDataset(train_ids, id_to_ecs_fold, cfg.DATA_PATHS["embedding_dir"])
    val_dataset_for_loss = ContrastiveDataset(val_ids, id_to_ecs_fold, cfg.DATA_PATHS["embedding_dir"])
    val_dataset_for_metrics = EvaluationDataset(val_ids, id_to_ecs_fold, cfg.DATA_PATHS["embedding_dir"])
    gallery_dataset = EvaluationDataset(gallery_ids, id_to_ecs_fold, cfg.DATA_PATHS["embedding_dir"])
    
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
    
    val_loader_loss = DataLoader(val_dataset_for_loss, batch_size=eval_batch_size, shuffle=False, collate_fn=contrastive_collate_fn, **cfg.DATALOADER_CONFIG)
    val_loader_for_metrics = DataLoader(val_dataset_for_metrics, batch_size=eval_batch_size, shuffle=False, collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)
    gallery_loader = DataLoader(gallery_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=evaluation_collate_fn, **cfg.DATALOCATION_CONFIG)

    # ==========================================================
    # --- 4. 初始化 & 运行 ---
    # ==========================================================
    
    logging.info("正在初始化模型、损失函数和优化器...")
    model = EcClassifier(**cfg.MODEL_CONFIG).to(cfg.DEVICE)
    criterion = SupervisedContrastiveLoss(temperature=cfg.TRAIN_CONFIG['temperature'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN_CONFIG['learning_rate'])
    
    runner = ExperimentRunner(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config={ 
            "TRAIN_CONFIG": cfg.TRAIN_CONFIG,
            "EVAL_CONFIG": cfg.EVAL_CONFIG,
            "TUNE_CONFIG": cfg.TUNE_CONFIG 
        },
        device=cfg.DEVICE,
        run_output_dir=run_output_dir 
    )
    
    logging.info("=== 开始最终模型训练 ===")
    try:
        runner.run(
            train_loader=train_loader,
            val_loader_loss=val_loader_loss,
            val_loader_metrics=val_loader_for_metrics,
            gallery_loader=gallery_loader,
            ood_test_loaders=ood_test_loaders 
        )
        
        logging.info("...最终模型训练完成。")
        
        final_results_path = os.path.join(run_output_dir, "final_evaluation_results.json")
        logging.info(f"最终的 OOD (PRICE, NEW) 评估结果已保存在: {final_results_path}")
        print(f"\n\n[成功]\n最终评估结果已保存: {final_results_path}")

    except Exception as e:
        logging.error("!!! 最终模型训练中遇到错误 !!!")
        logging.error(str(e), exc_info=True) 

    logging.info(f"实验 {run_name} 成功结束。")


if __name__ == '__main__':
    main()