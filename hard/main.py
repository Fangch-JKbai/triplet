# triplet/hard/main.py
"""
难负样本学习 - 主程序（支持外部测试集）
"""
import os

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import random
import numpy as np
import torch
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

# 导入项目模块（全部从本地导入，避免冲突）
import config as cfg
from dataset import TripletDataset, collate_fn
from sampler import ECDistanceAwareSampler
from model import EmbeddingProjector
from loss import AdaptiveTripletLoss, CircleLoss
from evaluator import Evaluator
from trainer import Trainer, create_scheduler
from data_utils import (
    load_and_create_label_map,
    augment_map_with_mutants,
    filter_ids_with_embeddings
)


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(output_dir):
    """设置日志"""
    log_path = os.path.join(output_dir, "experiment.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("Hard Negative Triplet Learning Experiment")
    logging.info("="*60)


def plot_pr_curves(pr_data_dict, output_dir):
    """绘制PR曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (name, (y_true, y_score, auprc)) in enumerate(pr_data_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, color=colors[idx % len(colors)],
                linewidth=2.5, label=f'{name} (AUPRC={auprc:.4f})')
    
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curves', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    save_path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"PR curves saved to {save_path}")


def main():
    # ========================================
    # 1. 初始化
    # ========================================
    set_seed(cfg.SEED)
    
    # 创建输出目录
    run_name = f"hard_negative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(cfg.OUTPUT_CONFIG["output_dir"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logging(output_dir)
    
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {cfg.DEVICE}")
    logging.info(f"Random seed: {cfg.SEED}")
    
    # ========================================
    # 2. 加载主数据集
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("Loading Main Dataset")
    logging.info("="*60)
    
    id_to_ecs_master, all_ids = load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    
    # ========================================
    # 3. 加载外部测试集标签（确保标签字典完整）
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("Loading External Test Sets")
    logging.info("="*60)
    
    external_test_ids = {}
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        logging.info(f"Loading {name} from {path}...")
        id_to_ecs_ext, ext_ids = load_and_create_label_map(path)
        
        # 合并到主标签映射中
        id_to_ecs_master.update(id_to_ecs_ext)
        
        external_test_ids[name] = ext_ids
        logging.info(f"  {name}: {len(ext_ids)} samples")
    
    # 只保留原始ID（不含下划线）
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    logging.info(f"\nMain dataset: {len(all_original_ids)} original samples")
    logging.info(f"Total unique ECs: {len(set([ec for ecs in id_to_ecs_master.values() for ec in ecs]))}")
    
    # ========================================
    # 4. 数据划分（只分train/val，无test）
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("Splitting Data into Train/Val (Random Split)")
    logging.info("="*60)
    
    val_ratio = cfg.DATA_CONFIG.get('val_split_ratio', 0.2)
    
    train_ids_orig, val_ids_orig = train_test_split(
        all_original_ids,
        test_size=val_ratio,
        random_state=cfg.SEED,
        stratify=None  # 随机划分，不分层
    )
    
    logging.info(f"\nData split (original sequences only):")
    logging.info(f"  Train: {len(train_ids_orig)} samples ({100*(1-val_ratio):.1f}%)")
    logging.info(f"  Val:   {len(val_ids_orig)} samples ({100*val_ratio:.1f}%)")
    
    # ========================================
    # 5. 添加突变序列（仅训练集）
    # ========================================
    if cfg.DATA_CONFIG.get("use_mutants", True):
        logging.info("\n" + "="*60)
        logging.info("Augmenting with Mutant Sequences")
        logging.info("="*60)
        
        # 增强映射表（添加突变体）
        id_to_ecs_master = augment_map_with_mutants(
            id_to_ecs_master,
            embedding_dir=cfg.DATA_PATHS["embedding_dir"],
            train_ids_orig_only=train_ids_orig
        )
        
        # 筛选出属于训练集的突变体
        mutants_train = [
            sid for sid in id_to_ecs_master.keys()
            if '_' in sid and sid.split('_')[0] in set(train_ids_orig)
        ]
        
        # 组合训练池（原始 + 突变体）
        train_pool_ids_raw = train_ids_orig + mutants_train
        
        logging.info(f"Training pool composition:")
        logging.info(f"  Original sequences: {len(train_ids_orig)}")
        logging.info(f"  Mutant sequences:   {len(mutants_train)}")
        logging.info(f"  Total:              {len(train_pool_ids_raw)}")
    else:
        train_pool_ids_raw = train_ids_orig
        logging.info("Mutants disabled, using only original sequences")
    
    # ========================================
    # 6. 过滤有embedding的样本
    # ========================================
    train_ids = filter_ids_with_embeddings(
        train_pool_ids_raw,
        cfg.DATA_PATHS["embedding_dir"],
        "[Train] "
    )
    
    val_ids = filter_ids_with_embeddings(
        val_ids_orig,
        cfg.DATA_PATHS["embedding_dir"],
        "[Val] "
    )
    
    # ========================================
    # 7. 创建Datasets
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("Creating Datasets")
    logging.info("="*60)
    
    train_dataset = TripletDataset(train_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    val_dataset = TripletDataset(val_ids, id_to_ecs_master, cfg.DATA_PATHS["embedding_dir"])
    
    # 创建外部测试集datasets
    external_test_datasets = {}
    for name, ext_ids in external_test_ids.items():
        logging.info(f"Creating dataset for {name}...")
        external_test_datasets[name] = TripletDataset(
            ext_ids, 
            id_to_ecs_master, 
            cfg.DATA_PATHS["embedding_dir"]
        )
    
    # ========================================
    # 8. 创建Sampler
    # ========================================
    sampler = ECDistanceAwareSampler(
        ec_to_indices=train_dataset.ec_to_indices,
        distance_dict_path=cfg.DATA_PATHS["distance_dict_path"],
        **cfg.SAMPLER_CONFIG
    )
    
    # ========================================
    # 9. 创建DataLoaders
    # ========================================
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        **cfg.DATALOADER_CONFIG
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.SAMPLER_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        **cfg.DATALOADER_CONFIG
    )
    
    # 创建外部测试集loaders
    external_test_loaders = {}
    for name, dataset in external_test_datasets.items():
        external_test_loaders[name] = DataLoader(
            dataset,
            batch_size=cfg.SAMPLER_CONFIG["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            **cfg.DATALOADER_CONFIG
        )
    
    # ========================================
    # 10. 创建模型和损失函数
    # ========================================
    model = EmbeddingProjector(**cfg.MODEL_CONFIG).to(cfg.DEVICE)
    
    if cfg.LOSS_CONFIG["loss_type"] == "adaptive_triplet":
        criterion = AdaptiveTripletLoss(
            margin_base=cfg.LOSS_CONFIG["margin_base"],
            margin_scale=cfg.LOSS_CONFIG["margin_scale"],
            distance_dict_path=cfg.DATA_PATHS["distance_dict_path"],
            distance_aware=cfg.LOSS_CONFIG["distance_aware"]
        )
    else:  # circle loss
        criterion = CircleLoss(
            m=cfg.LOSS_CONFIG["m"],
            gamma=cfg.LOSS_CONFIG["gamma"]
        )
    
    logging.info(f"\nModel: {model.__class__.__name__}")
    logging.info(f"Loss: {criterion.__class__.__name__}")
    
    # ========================================
    # 11. 创建优化器和调度器
    # ========================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.TRAIN_CONFIG["learning_rate"],
        weight_decay=cfg.TRAIN_CONFIG["weight_decay"]
    )
    
    scheduler = create_scheduler(optimizer, cfg.TRAIN_CONFIG)
    
    # ========================================
    # 12. 创建训练器和评估器
    # ========================================
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.DEVICE,
        config={
            "TRAIN_CONFIG": cfg.TRAIN_CONFIG,
            "EVAL_CONFIG": cfg.EVAL_CONFIG,
        },
        output_dir=output_dir
    )
    
    evaluator = Evaluator(model, cfg.DEVICE)
    
    # ========================================
    # 13. 训练循环
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("Starting Training")
    logging.info("="*60)
    
    for epoch in range(cfg.TRAIN_CONFIG["epochs"]):
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        trainer.history["learning_rate"].append(current_lr)
        
        # 训练
        train_loss = trainer.train_epoch(train_loader, sampler, epoch)
        trainer.history["train_loss"].append(train_loss)
        
        # 验证损失
        val_loss = trainer.evaluate_loss(val_loader)
        trainer.history["val_loss"].append(val_loss)
        
        logging.info(f"Epoch {epoch+1}/{cfg.TRAIN_CONFIG['epochs']} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.2e}")
        
        # 定期评估
        if (epoch + 1) % cfg.EVAL_CONFIG["eval_every_n_epochs"] == 0:
            logging.info("\n" + "-"*60)
            logging.info(f"Evaluation at Epoch {epoch+1}")
            logging.info("-"*60)
            
            # 提取embeddings
            train_data = evaluator.extract_embeddings(train_loader)
            val_data = evaluator.extract_embeddings(val_loader)
            
            # kNN评估
            knn_results, best_k = evaluator.evaluate_knn(
                train_data, val_data,
                k_values=cfg.EVAL_CONFIG["k_values"]
            )
            
            # 聚类质量
            if cfg.EVAL_CONFIG["compute_clustering_metrics"]:
                clustering_results = evaluator.evaluate_clustering(val_data)
            
            # 记录 F1（使用新方法）
            val_f1 = knn_results[best_k]["f1"]
            trainer.record_val_f1(epoch, val_f1)
            
            # 保存最佳模型
            if val_f1 > trainer.best_val_f1:
                improvement = val_f1 - trainer.best_val_f1
                trainer.best_val_f1 = val_f1
                trainer.best_epoch = epoch
                trainer.epochs_no_improve = 0
                trainer.save_checkpoint(epoch, is_best=True)
                
                logging.info(f"  ✓ New best F1: {val_f1:.4f} (improved by {improvement:.4f})")
            else:
                trainer.epochs_no_improve += 1
                logging.info(f"  ✗ No improvement for {trainer.epochs_no_improve} evaluation(s)")
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        # 早停检查
        if epoch >= cfg.TRAIN_CONFIG["early_stop_threshold_epoch"]:
            if trainer.epochs_no_improve >= cfg.TRAIN_CONFIG["patience"] // cfg.EVAL_CONFIG["eval_every_n_epochs"]:
                logging.info(f"\n{'='*60}")
                logging.info("Early stopping triggered")
                logging.info(f"Best F1: {trainer.best_val_f1:.4f} at epoch {trainer.best_epoch+1}")
                logging.info(f"{'='*60}\n")
                break
    
    # ========================================
    # 14. 最终评估（验证集 + 外部测试集）
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("Final Evaluation")
    logging.info("="*60)
    
    # 加载最佳模型
    best_model_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)
    else:
        logging.warning("Best model not found, using current model")
    
    # 提取训练集embeddings（作为gallery）
    train_data = evaluator.extract_embeddings(train_loader)
    
    # 评估验证集
    logging.info("\n" + "="*60)
    logging.info("Validation Set Evaluation")
    logging.info("="*60)
    
    val_data = evaluator.extract_embeddings(val_loader)
    val_knn_results, best_k = evaluator.evaluate_knn(
        train_data, val_data,
        k_values=cfg.EVAL_CONFIG["k_values"]
    )
    
    if cfg.EVAL_CONFIG["compute_clustering_metrics"]:
        val_clustering_results = evaluator.evaluate_clustering(val_data)
    else:
        val_clustering_results = {}
    
    # 收集PR曲线数据
    pr_data = {}
    if cfg.EVAL_CONFIG["plot_pr_curves"]:
        y_true_val, y_score_val, auprc_val = evaluator.compute_pr_curve_data(
            train_data, val_data, k=best_k
        )
        pr_data["Validation"] = (y_true_val, y_score_val, auprc_val)
    
    # 评估外部测试集
    external_test_results = {}
    
    for name, loader in external_test_loaders.items():
        logging.info("\n" + "="*60)
        logging.info(f"External Test Set: {name.upper()}")
        logging.info("="*60)
        
        test_data = evaluator.extract_embeddings(loader)
        test_knn_results, _ = evaluator.evaluate_knn(
            train_data, test_data,
            k_values=cfg.EVAL_CONFIG["k_values"]
        )
        
        if cfg.EVAL_CONFIG["compute_clustering_metrics"]:
            test_clustering_results = evaluator.evaluate_clustering(test_data)
        else:
            test_clustering_results = {}
        
        external_test_results[name] = {
            "knn_results": test_knn_results,
            "clustering_results": test_clustering_results
        }
        
        # 收集PR曲线数据
        if cfg.EVAL_CONFIG["plot_pr_curves"]:
            y_true_test, y_score_test, auprc_test = evaluator.compute_pr_curve_data(
                train_data, test_data, k=best_k
            )
            pr_data[name.upper()] = (y_true_test, y_score_test, auprc_test)
    
    # 绘制PR曲线
    if cfg.EVAL_CONFIG["plot_pr_curves"] and pr_data:
        plot_pr_curves(pr_data, output_dir)
    
    # ========================================
    # 15. 保存结果
    # ========================================
    final_results = {
        "best_epoch": trainer.best_epoch + 1,
        "best_val_f1": trainer.best_val_f1,
        "validation": {
            "knn_results": val_knn_results,
            "clustering_results": val_clustering_results,
            "auprc": pr_data.get("Validation", (None, None, 0.0))[2]
        },
        "external_test_sets": {}
    }
    
    # 添加外部测试集结果
    for name, results in external_test_results.items():
        final_results["external_test_sets"][name] = {
            "knn_results": results["knn_results"],
            "clustering_results": results["clustering_results"],
            "auprc": pr_data.get(name.upper(), (None, None, 0.0))[2]
        }
    
    trainer.save_results(final_results)
    trainer.plot_training_curves()
    
    # ========================================
    # 16. 打印最终总结
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("Experiment Completed Successfully!")
    logging.info("="*60)
    logging.info(f"Best Validation F1: {trainer.best_val_f1:.4f} at epoch {trainer.best_epoch+1}")
    logging.info(f"\nFinal Results (k={best_k}):")
    logging.info(f"  Validation F1: {val_knn_results[best_k]['f1']:.4f}")
    
    for name, results in external_test_results.items():
        f1 = results["knn_results"][best_k]["f1"]
        logging.info(f"  {name.upper()} F1: {f1:.4f}")
    
    if cfg.EVAL_CONFIG["compute_clustering_metrics"] and val_clustering_results:
        logging.info(f"\nClustering Quality:")
        logging.info(f"  Val Silhouette: {val_clustering_results.get('silhouette_score', 0):.4f}")
        for name, results in external_test_results.items():
            sil = results["clustering_results"].get('silhouette_score', 0)
            logging.info(f"  {name.upper()} Silhouette: {sil:.4f}")
    
    logging.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()