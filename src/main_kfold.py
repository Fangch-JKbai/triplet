# main_kfold.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import random
import numpy as np
import json
from sklearn.model_selection import KFold
from typing import List, Dict

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

def report_final_results(all_fold_results: List[Dict], base_output_dir: str):
    """
    在所有折叠完成后，聚合 OOD 测试集的结果并计算均值/标准差。
    """
    logging.info("="*30)
    logging.info("Cross-Validation Final Results Aggregation")
    logging.info("="*30)
    
    if not all_fold_results:
        logging.warning("No fold results found to aggregate.")
        return

    ood_sets = list(cfg.DATA_PATHS["external_test_sets"].keys()) 
    metric_prefixes = ["f1", "precision", "recall", "auc"]
    
    final_report = {}

    for ood_set in ood_sets:
        final_report[ood_set] = {}
        logging.info(f"--- OOD Set: {ood_set.upper()} ---")
        
        for prefix in metric_prefixes:
            scores = []
            for fold_result in all_fold_results:
                results_dict = fold_result.get("results", {}) 
                
                if ood_set in results_dict and isinstance(results_dict[ood_set], dict):
                    ood_set_dict = results_dict[ood_set]
                    actual_key_found = None # 每一折都要重置

                    if prefix == "auc":
                        if prefix in ood_set_dict:
                            actual_key_found = prefix # 键名就是 "auc"
                    else:
                        for key in ood_set_dict.keys():
                            if key.startswith(prefix + '_at_fixed_tau_'):
                                actual_key_found = key
                                break

                    if actual_key_found:
                        scores.append(ood_set_dict[actual_key_found])
                    # else: 
                    #   logging.debug(f"Could not find key {prefix} in {ood_set} for fold.")
            
            if scores:
                mean = np.mean(scores)
                std = np.std(scores)
                
                final_report[ood_set][prefix] = {"mean": mean, "std": std}
                
                log_msg = f"  {prefix.capitalize()}: {mean:.4f} ± {std:.4f}"
                logging.info(log_msg)
                print(f"[Final Report] {ood_set.upper()} - {prefix.capitalize()}: {mean:.4f} ± {std:.4f}")
            else:
                logging.warning(f"  No scores found for {ood_set} - (metric prefix {prefix})")

    # 将最终的聚合报告保存到主运行目录
    summary_path = os.path.join(base_output_dir, "final_cross_validation_summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(final_report, f, indent=4)
        logging.info(f"Final aggregated report saved to {summary_path}")
    except Exception as e:
        logging.error(f"Failed to save final aggregated report: {e}")


def main():
    # 定义实验名称
    run_name = cfg.ARTIFACTS_CONFIG.get('run_name')
    if run_name is None:
        run_name = datetime.now().strftime('%Y-%m-%d') + "_KFold"
    
    base_run_output_dir = os.path.join(cfg.ARTIFACTS_CONFIG['output_dir'], run_name)
    os.makedirs(base_run_output_dir, exist_ok=True)
    
    main_log_path = os.path.join(base_run_output_dir, "main_experiment.log")
    utils.setup_logging(main_log_path)
    
    set_seeds(cfg.SEED)

    logging.info(f"Experiment run: {run_name}")
    logging.info(f"Base artifacts directory: {base_run_output_dir}")
    logging.info(f"Using device: {cfg.DEVICE}")
    logging.info(f"Running {cfg.N_SPLITS}-Fold Cross-Validation.")

    logging.info("Preparing shared data (Master Label Map and OOD Loaders)...")
    
    # 2.1 加载所有标签信息到一个主映射中
    id_to_ecs_master_base, _ = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master_base.update(id_to_ecs_ext)
    
    # 2.2 获取所有用于交叉验证的原始ID
    _, all_main_ids = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    all_original_ids = [entry for entry in all_main_ids if '_' not in entry]
    all_original_ids_np = np.array(all_original_ids)
    
    logging.info(f"Total original IDs for K-Fold splitting: {len(all_original_ids_np)}")
    
    ood_test_loaders = {}
    eval_batch_size = cfg.TRAIN_CONFIG['P'] * cfg.TRAIN_CONFIG['K']
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        _, ext_test_ids = data_utils.load_and_create_label_map(path)
        ext_test_dataset = EvaluationDataset(ext_test_ids, id_to_ecs_master_base, cfg.DATA_PATHS["embedding_dir"], strict_embeddings=False)
        ood_test_loaders[name] = DataLoader(ext_test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)
    
    logging.info(f"Created {len(ood_test_loaders)} OOD test loaders.")

    # --- 3. K-Fold 交叉验证循环 ---
    kf = KFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
    
    all_fold_results = [] 

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_original_ids_np)):
        
        fold_name = f"fold_{fold_idx + 1}"
        logging.info("="*30)
        logging.info(f"STARTING {fold_name.upper()} ({fold_idx + 1}/{cfg.N_SPLITS})")
        logging.info("="*30)

        fold_output_dir = os.path.join(base_run_output_dir, fold_name)
        os.makedirs(fold_output_dir, exist_ok=True)
        
        fold_log_path = os.path.join(fold_output_dir, "fold.log")
        utils.setup_logging(fold_log_path) # 让日志记录器指向新的折叠文件
        logging.info(f"Artifacts for this fold will be saved to: {fold_output_dir}")

        # 3.2 <<< K-Fold 改动: 根据 KFold 索引获取当前折叠的 训练/验证 ID >>>
        train_ids_orig = all_original_ids_np[train_indices].tolist()
        val_ids_orig = all_original_ids_np[val_indices].tolist()
        
        logging.info(f"Fold Train Original IDs: {len(train_ids_orig)}")
        logging.info(f"Fold Validation Original IDs: {len(val_ids_orig)}")

        # 3.3 <<< K-Fold 改动: 数据增强和过滤 (在折叠内部完成) >>>
        id_to_ecs_fold = data_utils.augment_map_with_mutants(
            id_to_ecs_master_base.copy(), 
            embedding_dir=cfg.DATA_PATHS["embedding_dir"],
            train_ids_orig_only=train_ids_orig
        )
        
        mutants_train = [sid for sid in id_to_ecs_fold.keys() if '_' in sid and sid.split('_')[0] in set(train_ids_orig)]
        train_pool_ids_raw = train_ids_orig + mutants_train
        
        train_ids = data_utils.filter_ids_with_embeddings(train_pool_ids_raw, cfg.DATA_PATHS["embedding_dir"], f"[{fold_name} TrainPool] ")
        val_ids = data_utils.filter_ids_with_embeddings(val_ids_orig, cfg.DATA_PATHS["embedding_dir"], f"[{fold_name} Validation] ")

        gallery_ids = data_utils.filter_ids_with_embeddings(train_ids_orig, cfg.DATA_PATHS["embedding_dir"], f"[{fold_name} Gallery] ")

        # 3.4 <<< K-Fold 改动: 创建特定于此折叠的 Datasets 和 DataLoaders >>>
        logging.info("Creating fold-specific Datasets and DataLoaders...")
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
        gallery_loader = DataLoader(gallery_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG)

        # 3.5 <<< K-Fold 改动: 初始化全新的模型、损失和优化器 >>>
        logging.info("Initializing new model, criterion, and optimizer for this fold...")
        model = EcClassifier(**cfg.MODEL_CONFIG).to(cfg.DEVICE)
        criterion = SupervisedContrastiveLoss(temperature=cfg.TRAIN_CONFIG['temperature'])
        optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN_CONFIG['learning_rate'])
        
        # 3.6 <<< K-Fold 改动: 启动此折叠的 ExperimentRunner >>>
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
            run_output_dir=fold_output_dir
        )
        
        try:
            runner.run(
                train_loader=train_loader,
                val_loader_loss=val_loader_loss,
                val_loader_metrics=val_loader_for_metrics,
                gallery_loader=gallery_loader,
                ood_test_loaders=ood_test_loaders # OOD loaders 是共享的
            )
            
            # 3.7 <<< K-Fold 改动: 运行后，加载该折叠的结果以供最终聚合 >>>
            fold_results_path = os.path.join(fold_output_dir, "final_evaluation_results.json")
            if os.path.exists(fold_results_path):
                with open(fold_results_path, 'r') as f:
                    all_fold_results.append(json.load(f))
                logging.info(f"Successfully loaded results from {fold_name}.")
            else:
                logging.warning(f"Could not find results file for {fold_name} at {fold_results_path}")
                
        except Exception as e:
            logging.error(f"!!! ERROR encountered in {fold_name} !!!")
            logging.error(str(e), exc_info=True)

        logging.info(f"COMPLETED {fold_name.upper()}")
        
        # <<< K-Fold 改动: 将日志切换回“主”日志文件 >>>
        utils.setup_logging(main_log_path) 

    # --- 4. 循环结束，聚合所有结果 ---
    logging.info("All folds have completed.")
    report_final_results(all_fold_results, base_run_output_dir)
    logging.info(f"Experiment {run_name} finished successfully.")


if __name__ == '__main__':
    main()