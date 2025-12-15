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
from itertools import product
import json

import config as cfg
import data_utils
import utils
from model import EcClassifier
from datasets import ContrastiveDataset, EvaluationDataset, contrastive_collate_fn
from evaluation_v2 import evaluation_collate_fn, Evaluator
from loss import SupervisedContrastiveLoss
from samplers import PKSampler
from trainer import ExperimentRunner

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_single_experiment(hyperparams, base_output_dir, train_ids, val_ids, 
                          id_to_ecs_master, embedding_dir, external_test_sets, seed):
    """运行单次实验，返回验证集性能"""
    
    # 创建本次实验的输出目录
    exp_name = "_".join([f"{k}={v}" for k, v in hyperparams.items()])
    exp_output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(exp_output_dir, "experiment.log")
    utils.setup_logging(log_path)
    
    logging.info("="*60)
    logging.info(f"Running experiment with hyperparameters:")
    for k, v in hyperparams.items():
        logging.info(f"  {k}: {v}")
    logging.info("="*60)
    
    # 数据准备
    id_to_ecs_master_copy = id_to_ecs_master.copy()
    id_to_ecs_master_copy = data_utils.augment_map_with_mutants(
        id_to_ecs_master_copy,
        embedding_dir=embedding_dir,
        train_ids_orig_only=train_ids
    )
    
    mutants_train = [sid for sid in id_to_ecs_master_copy.keys() 
                     if '_' in sid and sid.split('_')[0] in set(train_ids)]
    train_pool_ids_raw = train_ids + mutants_train
    
    train_pool_ids = data_utils.filter_ids_with_embeddings(train_pool_ids_raw, embedding_dir, "[TrainPool] ")
    val_ids_filtered = data_utils.filter_ids_with_embeddings(val_ids, embedding_dir, "[Validation] ")
    gallery_ids = data_utils.filter_ids_with_embeddings(train_ids, embedding_dir, "[Gallery] ")
    
    # 创建数据集
    train_dataset = ContrastiveDataset(train_pool_ids, id_to_ecs_master_copy, embedding_dir)
    val_dataset_for_loss = ContrastiveDataset(val_ids_filtered, id_to_ecs_master_copy, embedding_dir)
    val_dataset_for_metrics = EvaluationDataset(val_ids_filtered, id_to_ecs_master_copy, embedding_dir)
    gallery_dataset = EvaluationDataset(gallery_ids, id_to_ecs_master_copy, embedding_dir)
    
    # 创建 DataLoaders
    pk_sampler = PKSampler(
        ec_to_indices=train_dataset.ec_to_indices,
        p=hyperparams['P'],
        k=hyperparams['K']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        collate_fn=contrastive_collate_fn,
        **cfg.DATALOADER_CONFIG
    )
    
    batch_size = hyperparams['P'] * hyperparams['K']
    val_loader_loss = DataLoader(val_dataset_for_loss, batch_size=batch_size, shuffle=False, 
                                  collate_fn=contrastive_collate_fn, **cfg.DATALOADER_CONFIG)
    val_loader_metrics = DataLoader(val_dataset_for_metrics, batch_size=batch_size, shuffle=False, 
                                    collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)
    
    # 创建模型
    model_config = cfg.MODEL_CONFIG.copy()
    model_config['dropout_rate'] = hyperparams['dropout_rate']
    model = EcClassifier(**model_config).to(cfg.DEVICE)
    
    criterion = SupervisedContrastiveLoss(temperature=hyperparams['temperature'])
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'])
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=cfg.TRAIN_CONFIG['epochs'],
        eta_min=cfg.TRAIN_CONFIG.get('scheduler_eta_min', 1e-6)
    )
    
    # 训练配置
    train_config = {
        "TRAIN_CONFIG": {
            **cfg.TRAIN_CONFIG,
            "P": hyperparams['P'],
            "K": hyperparams['K'],
            "temperature": hyperparams['temperature'],
            "learning_rate": hyperparams['learning_rate'],
            "patience": hyperparams['patience'],
        },
        "EVAL_CONFIG": cfg.EVAL_CONFIG,
        "TUNE_CONFIG": cfg.TUNE_CONFIG,
        "EXPERIMENT_MODE": "train_hyperparam_search"
    }
    
    # 运行训练
    runner = ExperimentRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=train_config,
        device=cfg.DEVICE,
        run_output_dir=exp_output_dir
    )
    
    # 简化版运行：只关心验证集损失
    runner.run(
        train_loader=train_loader,
        val_loader_loss=val_loader_loss,
        val_loader_metrics=val_loader_metrics,
        gallery_loader=gallery_loader,
        ood_test_loaders={}  # 不在超参数搜索时评估测试集
    )
    
    # 返回最佳验证损失
    return runner.best_val_loss, exp_output_dir

def main():
    set_seeds(cfg.SEED)
    
    # 创建超参数搜索的根目录
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    search_output_dir = os.path.join(cfg.ARTIFACTS_CONFIG['output_dir'], 
                                     f"hyperparam_search_{timestamp}")
    os.makedirs(search_output_dir, exist_ok=True)
    
    # 设置总日志
    summary_log_path = os.path.join(search_output_dir, "search_summary.log")
    utils.setup_logging(summary_log_path)
    
    logging.info("="*60)
    logging.info("Starting Hyperparameter Search")
    logging.info("="*60)
    
    # 数据准备
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master.update(id_to_ecs_ext)
    
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    # 划分训练集和验证集
    try:
        stratify_level = cfg.DATA_CONFIG.get('stratify_level', 3)
        labels_for_stratify = [
            '.'.join(id_to_ecs_master[uid][0].split('.')[:stratify_level]) 
            for uid in all_original_ids
        ]
    except Exception as e:
        logging.warning(f"无法创建分层标签: {e}")
        labels_for_stratify = None
    
    val_ratio = cfg.DATA_CONFIG.get('val_split_ratio', 0.2)
    train_ids, val_ids = train_test_split(
        all_original_ids,
        test_size=val_ratio,
        random_state=cfg.SEED,
        # stratify=labels_for_stratify
    )
    
    logging.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    # 生成所有超参数组合
    search_space = cfg.HYPERPARAM_SEARCH_SPACE
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(product(*values))
    
    logging.info(f"Total {len(combinations)} hyperparameter combinations to search")
    
    # 搜索
    results = []
    for i, combination in enumerate(combinations, 1):
        hyperparams = dict(zip(keys, combination))
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Experiment {i}/{len(combinations)}")
        logging.info(f"{'='*60}")
        
        try:
            best_val_loss, exp_dir = run_single_experiment(
                hyperparams=hyperparams,
                base_output_dir=search_output_dir,
                train_ids=train_ids,
                val_ids=val_ids,
                id_to_ecs_master=id_to_ecs_master,
                embedding_dir=cfg.DATA_PATHS["embedding_dir"],
                external_test_sets=cfg.DATA_PATHS["external_test_sets"],
                seed=cfg.SEED
            )
            
            result = {
                'hyperparams': hyperparams,
                'best_val_loss': best_val_loss,
                'exp_dir': exp_dir
            }
            results.append(result)
            
            logging.info(f"Completed: Val Loss = {best_val_loss:.4f}")
            
        except Exception as e:
            logging.error(f"Experiment failed with error: {e}")
            continue
    
    # 保存结果摘要
    results_sorted = sorted(results, key=lambda x: x['best_val_loss'])
    
    summary_path = os.path.join(search_output_dir, "search_results.json")
    with open(summary_path, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    
    # 打印最佳结果
    logging.info("\n" + "="*60)
    logging.info("HYPERPARAMETER SEARCH COMPLETED")
    logging.info("="*60)
    logging.info("\nTop 5 Results:")
    for i, result in enumerate(results_sorted[:5], 1):
        logging.info(f"\n{i}. Val Loss: {result['best_val_loss']:.4f}")
        logging.info(f"   Hyperparams: {result['hyperparams']}")
    
    logging.info(f"\nFull results saved to: {summary_path}")
    logging.info(f"\nBest hyperparameters:")
    for k, v in results_sorted[0]['hyperparams'].items():
        logging.info(f"  {k}: {v}")

if __name__ == '__main__':
    main()