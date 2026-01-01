# main_improved.py
"""
改进版主程序：全面评估embedding质量

主要改进：
1. 支持线性/MLP两种模型架构
2. 更大的超参数搜索空间
3. Embedding内在质量评估（聚类质量、k-NN、相似度分布等）
4. 按EC频率分层评估
5. 丰富的可视化（t-SNE、热图、分布图等）
6. 综合报告生成
"""
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
from collections import defaultdict
# 导入改进版模块
import config_improved as cfg
import data_utils
import utils
from model_improved import EcClassifier
from datasets import ContrastiveDataset, EvaluationDataset, contrastive_collate_fn
from evaluation_v3 import evaluation_collate_fn
from loss import SupervisedContrastiveLoss
from triplet.src.pk_sampler_v1_backup import PKSampler
from trainer_v3 import ComprehensiveExperimentRunner

def set_seeds(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def report_final_results(all_fold_results: List[Dict], base_output_dir: str):
    """
    聚合K-Fold结果并生成最终报告（修复版 - 兼容实际数据结构）
    """
    logging.info("="*60)
    logging.info("Cross-Validation Final Results Aggregation")
    logging.info("="*60)
    
    if not all_fold_results:
        logging.warning("No fold results found.")
        return
    
    ood_sets = list(cfg.DATA_PATHS["external_test_sets"].keys())
    final_report = {}
    
    # 对每个OOD集合聚合结果
    for ood_set in ood_sets:
        final_report[ood_set] = {}
        logging.info(f"\n--- OOD Set: {ood_set.upper()} ---")
        
        # 收集所有fold的结果
        all_metrics = defaultdict(list)
        
        for fold_result in all_fold_results:
            # ✅ 修复：直接从fold_result获取results
            if isinstance(fold_result, dict):
                if "results" in fold_result:
                    results_dict = fold_result["results"]
                else:
                    # 兼容：如果fold_result本身就是results
                    results_dict = fold_result
            else:
                continue
            
            # 下游任务指标（从ood_set键下获取）
            if ood_set in results_dict:
                ood_dict = results_dict[ood_set]
                for key, value in ood_dict.items():
                    if isinstance(value, (int, float)):
                        all_metrics[key].append(value)
            
            # 内在质量指标（从ood_set_intrinsic键下获取）
            intrinsic_key = f"{ood_set}_intrinsic"
            if intrinsic_key in results_dict:
                intrinsic_dict = results_dict[intrinsic_key]
                for key, value in intrinsic_dict.items():
                    if isinstance(value, (int, float)):
                        # 加前缀避免重名
                        all_metrics[f"intrinsic_{key}"].append(value)
        
        # 计算mean和std
        for metric_name, values in all_metrics.items():
            if values:
                mean = np.mean(values)
                std = np.std(values)
                final_report[ood_set][metric_name] = {"mean": mean, "std": std}
                logging.info(f"  {metric_name}: {mean:.4f} ± {std:.4f}")
            else:
                logging.warning(f"  No data for {metric_name}")
        
        if not all_metrics:
            logging.warning(f"  No metrics found for {ood_set}")
    
    # 保存JSON
    summary_path = os.path.join(base_output_dir, "final_cross_validation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(final_report, f, indent=4)
    logging.info(f"\nFinal report saved to {summary_path}")
    
    # 生成Markdown报告
    _generate_markdown_report(final_report, base_output_dir)

def _generate_markdown_report(report: Dict, output_dir: str):
    """生成Markdown格式报告（修复版）"""
    md_path = os.path.join(output_dir, "FINAL_REPORT.md")
    
    with open(md_path, 'w') as f:
        f.write("# Comprehensive Embedding Quality Evaluation Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        for ood_set, metrics in report.items():
            f.write(f"## {ood_set.upper()} Test Set\n\n")
            
            if not metrics:
                f.write("*No data available for this test set.*\n\n")
                continue
            
            # 下游任务性能
            downstream_keys = [k for k in metrics.keys() 
                             if not k.startswith('intrinsic_')]
            if downstream_keys:
                f.write("### Downstream Task Performance\n\n")
                f.write("| Metric | Mean | Std |\n")
                f.write("|--------|------|-----|\n")
                for key in sorted(downstream_keys):
                    mean = metrics[key]['mean']
                    std = metrics[key]['std']
                    # 美化指标名
                    display_name = key.replace('_at_fixed_tau_0.35', '').replace('_', ' ').upper()
                    f.write(f"| {display_name} | {mean:.4f} | {std:.4f} |\n")
                f.write("\n")
            
            # 内在质量
            intrinsic_keys = [k for k in metrics.keys() 
                            if k.startswith('intrinsic_')]
            if intrinsic_keys:
                f.write("### Intrinsic Embedding Quality\n\n")
                f.write("| Metric | Mean | Std |\n")
                f.write("|--------|------|-----|\n")
                for key in sorted(intrinsic_keys):
                    mean = metrics[key]['mean']
                    std = metrics[key]['std']
                    # 去掉intrinsic_前缀并美化
                    display_name = key.replace('intrinsic_', '').replace('_', ' ').title()
                    f.write(f"| {display_name} | {mean:.4f} | {std:.4f} |\n")
                f.write("\n")
            
            f.write("---\n\n")
    
    logging.info(f"Markdown report saved to {md_path}")

def main():
    # =============================================================
    # 1. 实验环境设置
    # =============================================================
    run_name = cfg.ARTIFACTS_CONFIG.get('run_name')
    if run_name is None:
        run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f"_KFold_{cfg.COMPARISON_MODE}"
    
    base_run_output_dir = os.path.join(cfg.ARTIFACTS_CONFIG['output_dir'], run_name)
    os.makedirs(base_run_output_dir, exist_ok=True)
    
    main_log_path = os.path.join(base_run_output_dir, "main_experiment.log")
    utils.setup_logging(main_log_path)
    
    set_seeds(cfg.SEED)
    
    logging.info("="*60)
    logging.info(f"Experiment: {run_name}")
    logging.info(f"Mode: {cfg.COMPARISON_MODE}")
    logging.info(f"Device: {cfg.DEVICE}")
    logging.info(f"K-Fold: {cfg.N_SPLITS} splits")
    logging.info("="*60)
    
    # =============================================================
    # 2. 数据准备
    # =============================================================
    logging.info("\nPreparing data...")
    
    # 加载标签映射
    id_to_ecs_master_base, _ = data_utils.load_and_create_label_map(
        cfg.DATA_PATHS["csv_path"]
    )
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master_base.update(id_to_ecs_ext)
    
    # 获取所有原始ID用于K-Fold
    _, all_main_ids = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    all_original_ids = [entry for entry in all_main_ids if '_' not in entry]
    all_original_ids_np = np.array(all_original_ids)
    
    logging.info(f"Total original IDs: {len(all_original_ids_np)}")
    
    # 创建OOD测试集loaders（所有fold共享）
    ood_test_loaders = {}
    eval_batch_size = cfg.TRAIN_CONFIG['P'] * cfg.TRAIN_CONFIG['K']
    
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        _, ext_test_ids = data_utils.load_and_create_label_map(path)
        ext_test_dataset = EvaluationDataset(
            ext_test_ids, id_to_ecs_master_base, 
            cfg.DATA_PATHS["embedding_dir"], strict_embeddings=False
        )
        ood_test_loaders[name] = DataLoader(
            ext_test_dataset, batch_size=eval_batch_size, 
            shuffle=False, collate_fn=evaluation_collate_fn, 
            **cfg.DATALOADER_CONFIG
        )
    
    logging.info(f"Created {len(ood_test_loaders)} OOD test loaders.")
    
    # =============================================================
    # 3. K-Fold交叉验证
    # =============================================================
    kf = KFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
    all_fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_original_ids_np)):
        fold_name = f"fold_{fold_idx + 1}"
        logging.info("\n" + "="*60)
        logging.info(f"FOLD {fold_idx + 1}/{cfg.N_SPLITS}")
        logging.info("="*60)
        
        # 创建fold目录
        fold_output_dir = os.path.join(base_run_output_dir, fold_name)
        os.makedirs(fold_output_dir, exist_ok=True)
        
        fold_log_path = os.path.join(fold_output_dir, "fold.log")
        utils.setup_logging(fold_log_path)
        
        # 数据划分
        train_ids_orig = all_original_ids_np[train_indices].tolist()
        val_ids_orig = all_original_ids_np[val_indices].tolist()
        
        logging.info(f"Train samples: {len(train_ids_orig)}")
        logging.info(f"Val samples: {len(val_ids_orig)}")
        
        # 数据增强（基于当前fold的训练集）
        id_to_ecs_fold = data_utils.augment_map_with_mutants(
            id_to_ecs_master_base.copy(),
            embedding_dir=cfg.DATA_PATHS["embedding_dir"],
            train_ids_orig_only=train_ids_orig
        )
        
        mutants_train = [sid for sid in id_to_ecs_fold.keys() 
                        if '_' in sid and sid.split('_')[0] in set(train_ids_orig)]
        train_pool_ids_raw = train_ids_orig + mutants_train
        
        train_ids = data_utils.filter_ids_with_embeddings(
            train_pool_ids_raw, cfg.DATA_PATHS["embedding_dir"], 
            f"[{fold_name} Train] "
        )
        val_ids = data_utils.filter_ids_with_embeddings(
            val_ids_orig, cfg.DATA_PATHS["embedding_dir"], 
            f"[{fold_name} Val] "
        )
        gallery_ids = data_utils.filter_ids_with_embeddings(
            train_ids_orig, cfg.DATA_PATHS["embedding_dir"], 
            f"[{fold_name} Gallery] "
        )
        
        # 创建数据集和loaders
        train_dataset = ContrastiveDataset(
            train_ids, id_to_ecs_fold, cfg.DATA_PATHS["embedding_dir"]
        )
        val_dataset_loss = ContrastiveDataset(
            val_ids, id_to_ecs_fold, cfg.DATA_PATHS["embedding_dir"]
        )
        val_dataset_metrics = EvaluationDataset(
            val_ids, id_to_ecs_fold, cfg.DATA_PATHS["embedding_dir"]
        )
        gallery_dataset = EvaluationDataset(
            gallery_ids, id_to_ecs_fold, cfg.DATA_PATHS["embedding_dir"]
        )
        
        pk_sampler = PKSampler(
            ec_to_indices=train_dataset.ec_to_indices,
            p=cfg.TRAIN_CONFIG['P'],
            k=cfg.TRAIN_CONFIG['K']
        )
        
        train_loader = DataLoader(
            train_dataset, batch_sampler=pk_sampler,
            collate_fn=contrastive_collate_fn, **cfg.DATALOADER_CONFIG
        )
        val_loader_loss = DataLoader(
            val_dataset_loss, batch_size=eval_batch_size, 
            shuffle=False, collate_fn=contrastive_collate_fn, 
            **cfg.DATALOADER_CONFIG
        )
        val_loader_metrics = DataLoader(
            val_dataset_metrics, batch_size=eval_batch_size, 
            shuffle=False, collate_fn=evaluation_collate_fn, 
            **cfg.DATALOADER_CONFIG
        )
        gallery_loader = DataLoader(
            gallery_dataset, batch_size=eval_batch_size, 
            shuffle=False, collate_fn=evaluation_collate_fn, 
            **cfg.DATALOADER_CONFIG
        )
        
        # 初始化模型（每个fold独立初始化）
        logging.info(f"Initializing model (architecture: {cfg.MODEL_CONFIG['architecture']})...")
        model = EcClassifier(**cfg.MODEL_CONFIG).to(cfg.DEVICE)
        criterion = SupervisedContrastiveLoss(temperature=cfg.TRAIN_CONFIG['temperature'])
        optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN_CONFIG['learning_rate'])
        
        # 创建Runner并运行
        runner = ComprehensiveExperimentRunner(
            model=model, optimizer=optimizer, criterion=criterion,
            config={
                "TRAIN_CONFIG": cfg.TRAIN_CONFIG,
                "EVAL_CONFIG": cfg.EVAL_CONFIG,
                "TUNE_CONFIG": cfg.TUNE_CONFIG,
                "INTRINSIC_EVAL_CONFIG": cfg.INTRINSIC_EVAL_CONFIG,
                "VISUALIZATION_CONFIG": cfg.VISUALIZATION_CONFIG
            },
            device=cfg.DEVICE,
            run_output_dir=fold_output_dir
        )
        
        try:
            runner.run(
                train_loader=train_loader,
                val_loader_loss=val_loader_loss,
                val_loader_metrics=val_loader_metrics,
                gallery_loader=gallery_loader,
                ood_test_loaders=ood_test_loaders
            )
            
            # 加载结果
            # 3.7 运行后，加载该折叠的结果
            fold_results_path = os.path.join(fold_output_dir, "final_evaluation_results.json")
            if os.path.exists(fold_results_path):
                with open(fold_results_path, 'r') as f:
                    fold_data = json.load(f)
                    all_fold_results.append({
                        "fold": fold_name,
                        "results": fold_data.get("results", fold_data)  # 兼容两种格式
                    })
                logging.info(f"Successfully loaded results from {fold_name}.")
            else:
                logging.warning(f"Could not find results file for {fold_name}")
        
        except Exception as e:
            logging.error(f"ERROR in {fold_name}: {str(e)}", exc_info=True)
        
        # 切回主日志
        utils.setup_logging(main_log_path)
    
    # =============================================================
    # 4. 聚合所有fold结果
    # =============================================================
    logging.info("\n" + "="*60)
    logging.info("All folds completed. Aggregating results...")
    logging.info("="*60)
    
    report_final_results(all_fold_results, base_run_output_dir)
    
    logging.info("\n" + "="*60)
    logging.info(f"EXPERIMENT '{run_name}' FINISHED SUCCESSFULLY!")
    logging.info("="*60)
    logging.info(f"All artifacts saved to: {base_run_output_dir}")

if __name__ == '__main__':
    main()