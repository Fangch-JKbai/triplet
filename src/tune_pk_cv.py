#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tune_pk_cv.py - Stage A: Cluster-aware 5-fold CV for P/K selection

用法示例:
    python tune_pk_cv.py --model esm --cluster_tsv /path/to/cluster.tsv --pk_list 64x4 32x8 16x16 --epochs 500

输出:
    experiments/tune/<model>/pk_cv_results.csv
    experiments/tune/<model>/best_pk.json
"""

import os
import sys
import json
import argparse
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# 项目模块
import config as cfg
import data_utils
from model import EcClassifier
from datasets import EvaluationDataset
from evaluation import evaluation_collate_fn, Evaluator
from main import train_one_run, set_seeds


# ============================================================================
# Embedding 路径映射
# ============================================================================
EMBEDDING_PATHS = {
    "esm": "/home/fangchh/workdir/triplet/data/ESM_2/protein_embeddings",
    "cpt": "/home/fangchh/workdir/triplet/data/ESM_CPT/protein_embeddings",
    "sub": "/home/fangchh/workdir/triplet/data/ESM_SUB/protein_embeddings",
    "cpt_sub": "/home/fangchh/workdir/triplet/data/ESM_CPT_SUB/protein_embeddings",
}


# ============================================================================
# A) 解析聚类 TSV
# ============================================================================
def parse_cluster_tsv(cluster_tsv_path: str) -> Dict[str, str]:
    """
    解析 mmseqs 聚类 TSV 文件，构建 member -> cluster_id (rep) 映射。
    
    TSV 格式: rep<TAB>member
    每行表示 member 属于以 rep 为代表的 cluster。
    
    Args:
        cluster_tsv_path: 聚类文件路径
        
    Returns:
        member_to_cluster: dict, member_seq_id -> cluster_id (rep_seq_id)
    """
    member_to_cluster = {}
    
    with open(cluster_tsv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            rep, member = parts[0], parts[1]
            member_to_cluster[member] = rep
    
    logging.info(f"Parsed cluster TSV: {len(member_to_cluster)} member->cluster mappings")
    
    # 统计 cluster 数量
    unique_clusters = set(member_to_cluster.values())
    logging.info(f"Total unique clusters: {len(unique_clusters)}")
    
    return member_to_cluster


def get_cluster_ids_for_dataset(
    seq_ids: List[str],
    member_to_cluster: Dict[str, str]
) -> Tuple[List[str], int]:
    """
    为数据集中的每个 seq_id 获取 cluster_id。
    
    Args:
        seq_ids: 数据集中的序列 ID 列表
        member_to_cluster: member -> cluster_id 映射
        
    Returns:
        cluster_ids: 每个 seq_id 对应的 cluster_id 列表
        n_missing: 缺失 cluster 映射的数量
    """
    cluster_ids = []
    n_missing = 0
    
    for seq_id in seq_ids:
        # 如果是突变体 (含 '_')，先尝试用原始 ID 查找
        base_id = seq_id.split('_')[0] if '_' in seq_id else seq_id
        
        if seq_id in member_to_cluster:
            cluster_ids.append(member_to_cluster[seq_id])
        elif base_id in member_to_cluster:
            cluster_ids.append(member_to_cluster[base_id])
        else:
            # 单独成簇
            cluster_ids.append(seq_id)
            n_missing += 1
    
    if n_missing > 0:
        logging.warning(f"Found {n_missing}/{len(seq_ids)} seq_ids without cluster mapping (assigned to self-cluster)")
    
    return cluster_ids, n_missing


# ============================================================================
# B) Cluster-aware 5-fold CV 评估 (内存优化版)
# ============================================================================

import torch.nn.functional as F
from torch.cuda.amp import autocast


@torch.no_grad()
def _get_embeddings_from_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    normalize: bool = True,
    desc: str = "Encoding"
) -> Tuple[torch.Tensor, List[List[str]], List[str]]:
    """
    从 DataLoader 提取所有 embeddings（存放在 CPU 上以节省显存）
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_ids = []
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        if isinstance(batch, dict) and batch.get("empty", False):
            continue
        
        emb_1280 = batch['embedding_1280'].to(device, non_blocking=True)
        
        with autocast(enabled=(device.type == 'cuda')):
            emb_128 = model(emb_1280)
        
        if normalize:
            emb_128 = F.normalize(emb_128, dim=1)
        
        # 立即移到 CPU 释放显存
        all_embeddings.append(emb_128.cpu())
        all_labels.extend(batch['labels'])
        all_ids.extend(batch['seq_ids'])
    
    if not all_embeddings:
        return torch.empty(0, 128), [], []
    
    return torch.cat(all_embeddings, dim=0), all_labels, all_ids


@torch.no_grad()
def _chunked_knn_evaluate(
    query_emb: torch.Tensor,      # (Q, D) on CPU
    query_labels: List[List[str]],
    gallery_emb: torch.Tensor,    # (G, D) on CPU
    gallery_labels: List[List[str]],
    device: torch.device,
    k: int,
    tau: float,
    weighted: bool = True,
    chunk_size: int = 1024,       # 每次处理的 query 数量
) -> Dict[str, float]:
    """
    分块计算 kNN 评估，避免 OOM。
    
    策略：
    - Gallery embeddings 放在 GPU 上（一次性）
    - Query embeddings 分块送入 GPU 计算相似度
    """
    Q, D = query_emb.shape
    G = gallery_emb.shape[0]
    
    # Gallery 放到 GPU（如果 gallery 太大也会 OOM，但通常可以接受）
    # 如果 gallery 也太大，可以进一步分块，但这里先简化处理
    gallery_gpu = gallery_emb.to(device)
    
    # 构建 label vocabulary
    label_to_idx = {}
    idx_to_label = []
    for labels in gallery_labels + query_labels:
        for lb in labels:
            if lb not in label_to_idx:
                label_to_idx[lb] = len(idx_to_label)
                idx_to_label.append(lb)
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    # 分块处理 query
    for start_idx in range(0, Q, chunk_size):
        end_idx = min(start_idx + chunk_size, Q)
        
        # 当前 chunk 的 query
        query_chunk = query_emb[start_idx:end_idx].to(device)
        chunk_labels = query_labels[start_idx:end_idx]
        
        # 计算相似度 (chunk_size, G)
        sims = torch.matmul(query_chunk.float(), gallery_gpu.T.float())
        
        # Top-k
        k_eff = min(k, G)
        topk_sims, topk_idx = sims.topk(k_eff, largest=True, dim=1)
        
        # 转到 CPU 处理投票逻辑
        topk_sims = topk_sims.cpu()
        topk_idx = topk_idx.cpu()
        
        # 对这个 chunk 的每个 query 进行投票
        for i in range(len(chunk_labels)):
            true_set = set(chunk_labels[i])
            
            # 投票
            counter = defaultdict(float)
            if weighted:
                w_vec = torch.softmax(topk_sims[i].float(), dim=0)
                for jpos, idx in enumerate(topk_idx[i].tolist()):
                    w = float(w_vec[jpos].item())
                    for ec in gallery_labels[idx]:
                        counter[ec] += w
                thr_ref = 1.0  # softmax 和为 1
            else:
                for idx in topk_idx[i].tolist():
                    for ec in gallery_labels[idx]:
                        counter[ec] += 1.0
                thr_ref = float(k_eff)
            
            # 阈值预测
            thr = tau * thr_ref
            pred_set = {ec for ec, score in counter.items() if score >= thr}
            
            if not pred_set and counter:
                pred_set = {max(counter.items(), key=lambda x: x[1])[0]}
            
            # 计算指标
            tp = len(true_set & pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec = tp / len(true_set) if true_set else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            f1_scores.append(f1)
            precision_scores.append(prec)
            recall_scores.append(rec)
        
        # 释放 chunk 显存
        del query_chunk, sims, topk_sims, topk_idx
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 释放 gallery
    del gallery_gpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
        'precision': float(np.mean(precision_scores)) if precision_scores else 0.0,
        'recall': float(np.mean(recall_scores)) if recall_scores else 0.0,
        'auc': 0.0,  # 简化版不计算 AUC
        'auc_macro': 0.0,
    }


def cluster_aware_cv_evaluate(
    model: torch.nn.Module,
    dataset: EvaluationDataset,
    cluster_ids: List[str],
    device: torch.device,
    eval_config: Dict,
    k: int = 3,
    tau: float = 0.35,
    n_splits: int = 5,
    seed: int = 42,
    batch_size: int = 256,
    chunk_size: int = 1024,  # 新增：分块大小
) -> Dict[str, float]:
    """
    使用 GroupKFold 进行 cluster-aware 5-fold CV 评估（内存优化版）。
    
    每折:
    - gallery = 4 折数据 (train_idx)
    - query = 1 折数据 (val_idx)
    - 严格避免 query 进入 gallery
    
    Args:
        model: 训练好的模型
        dataset: EvaluationDataset
        cluster_ids: 每个样本对应的 cluster_id
        device: 计算设备
        eval_config: 评估配置
        k: kNN 的 k 值
        tau: 投票阈值
        n_splits: CV 折数
        seed: 随机种子
        batch_size: batch size for encoding
        chunk_size: chunk size for similarity computation
        
    Returns:
        results: dict with mean/std of metrics
    """
    set_seeds(seed)
    
    # 将 cluster_id 转换为整数编码（GroupKFold 需要）
    unique_clusters = sorted(set(cluster_ids))
    cluster_to_int = {c: i for i, c in enumerate(unique_clusters)}
    groups = np.array([cluster_to_int[c] for c in cluster_ids])
    
    logging.info(f"CV setup: {len(dataset)} samples, {len(unique_clusters)} clusters, {n_splits} folds")
    logging.info(f"Memory optimization: chunk_size={chunk_size}")
    
    # GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_metrics = {
        'f1': [], 'precision': [], 'recall': [], 'auc': [], 'auc_macro': []
    }
    
    indices = np.arange(len(dataset))
    weighted = eval_config.get('weighted', True)
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        logging.info(f"\n{'='*40}")
        logging.info(f"Fold {fold_idx + 1}/{n_splits}")
        logging.info(f"  Gallery (train): {len(train_idx)} samples")
        logging.info(f"  Query (val): {len(val_idx)} samples")
        
        # 创建 gallery 和 query 的 Subset
        gallery_subset = Subset(dataset, train_idx.tolist())
        query_subset = Subset(dataset, val_idx.tolist())
        
        # 创建 DataLoader
        gallery_loader = DataLoader(
            gallery_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=evaluation_collate_fn,
            num_workers=cfg.DATALOADER_CONFIG.get('num_workers', 4),
            pin_memory=cfg.DATALOADER_CONFIG.get('pin_memory', True),
        )
        
        query_loader = DataLoader(
            query_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=evaluation_collate_fn,
            num_workers=cfg.DATALOADER_CONFIG.get('num_workers', 4),
            pin_memory=cfg.DATALOADER_CONFIG.get('pin_memory', True),
        )
        
        # 提取 embeddings（存在 CPU 上）
        gallery_emb, gallery_labels, gallery_ids = _get_embeddings_from_loader(
            model, gallery_loader, device, normalize=True, desc=f"Fold {fold_idx+1} Gallery"
        )
        query_emb, query_labels, query_ids = _get_embeddings_from_loader(
            model, query_loader, device, normalize=True, desc=f"Fold {fold_idx+1} Query"
        )
        
        logging.info(f"  Gallery embeddings: {gallery_emb.shape}")
        logging.info(f"  Query embeddings: {query_emb.shape}")
        
        # 分块 kNN 评估
        metrics = _chunked_knn_evaluate(
            query_emb=query_emb,
            query_labels=query_labels,
            gallery_emb=gallery_emb,
            gallery_labels=gallery_labels,
            device=device,
            k=k,
            tau=tau,
            weighted=weighted,
            chunk_size=chunk_size,
        )
        
        # 释放内存
        del gallery_emb, query_emb
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 记录指标
        for metric_name in fold_metrics.keys():
            if metric_name in metrics:
                fold_metrics[metric_name].append(metrics[metric_name])
        
        logging.info(f"  Fold {fold_idx + 1} results: F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    
    # 计算 mean ± std
    results = {}
    for metric_name, values in fold_metrics.items():
        if values:
            results[f'mean_{metric_name}'] = float(np.mean(values))
            results[f'std_{metric_name}'] = float(np.std(values))
    
    logging.info(f"\n{'='*40}")
    logging.info(f"CV Summary (k={k}, tau={tau}):")
    logging.info(f"  F1:        {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    logging.info(f"  Precision: {results['mean_precision']:.4f} ± {results['std_precision']:.4f}")
    logging.info(f"  Recall:    {results['mean_recall']:.4f} ± {results['std_recall']:.4f}")
    
    return results


# ============================================================================
# C) P/K 搜索主逻辑
# ============================================================================
def parse_pk_list(pk_str_list: List[str]) -> List[Tuple[int, int]]:
    """解析 P/K 列表，如 ['64x4', '32x8'] -> [(64, 4), (32, 8)]"""
    result = []
    for pk_str in pk_str_list:
        parts = pk_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid P/K format: {pk_str}. Expected format: PxK (e.g., 64x4)")
        p, k = int(parts[0]), int(parts[1])
        if p * k != 256:
            logging.warning(f"P*K = {p*k} != 256 for {pk_str}")
        result.append((p, k))
    return result


def run_pk_tuning(
    model_name: str,
    cluster_tsv_path: str,
    pk_list: List[Tuple[int, int]],
    epochs: int,
    seed: int,
    cv_k: int = 3,
    cv_tau: float = 0.35,
    n_splits: int = 5,
):
    """
    P/K 调参主函数。
    
    对每个 (P,K):
    1. 调用 train_one_run 训练模型
    2. 使用 cluster-aware 5-fold CV 评估
    3. 记录结果
    
    最终选择 mean_f1 最高的 (P,K)
    """
    # 获取 embedding 路径
    if model_name not in EMBEDDING_PATHS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(EMBEDDING_PATHS.keys())}")
    embedding_dir = EMBEDDING_PATHS[model_name]
    
    # 输出目录
    output_base = os.path.join("experiments", "tune", model_name)
    os.makedirs(output_base, exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(output_base, f"pk_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置 logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*60)
    logging.info(f"P/K Tuning for model: {model_name}")
    logging.info(f"Embedding dir: {embedding_dir}")
    logging.info(f"Cluster TSV: {cluster_tsv_path}")
    logging.info(f"P/K candidates: {pk_list}")
    logging.info(f"Training epochs: {epochs}")
    logging.info(f"CV settings: k={cv_k}, tau={cv_tau}, n_splits={n_splits}, seed={seed}")
    logging.info("="*60)
    
    # 解析聚类文件
    member_to_cluster = parse_cluster_tsv(cluster_tsv_path)
    
    # 加载数据集（用于 CV 评估）
    csv_path = cfg.DATA_PATHS["csv_path"]
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(csv_path)
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    # 增强标签映射（包含突变体）
    id_to_ecs_master = data_utils.augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=embedding_dir,
        train_ids_orig_only=all_original_ids
    )
    
    # 获取训练池 IDs
    mutants_train = [sid for sid in id_to_ecs_master.keys() 
                    if '_' in sid and sid.split('_')[0] in set(all_original_ids)]
    train_pool_ids_raw = all_original_ids + mutants_train
    train_ids = data_utils.filter_ids_with_embeddings(train_pool_ids_raw, embedding_dir, "[TrainPool] ")
    
    # 创建评估数据集
    eval_dataset = EvaluationDataset(train_ids, id_to_ecs_master, embedding_dir)
    
    # 获取每个样本的 seq_id 和对应的 cluster_id
    seq_ids = [eval_dataset.ids[i] for i in range(len(eval_dataset))]
    cluster_ids, n_missing = get_cluster_ids_for_dataset(seq_ids, member_to_cluster)
    
    logging.info(f"Prepared CV dataset: {len(eval_dataset)} samples")
    logging.info(f"Cluster mapping: {len(cluster_ids)} IDs, {n_missing} missing")
    
    # 存储所有结果
    all_results = []
    device = cfg.DEVICE
    
    for p, k in pk_list:
        logging.info("\n" + "="*60)
        logging.info(f"Training with P={p}, K={k}")
        logging.info("="*60)
        
        # 训练目录
        train_output_dir = os.path.join(output_base, f"P{p}_K{k}")
        
        # 训练模型
        try:
            best_model_path = train_one_run(
                embedding_dir=embedding_dir,
                P=p,
                K=k,
                output_dir=train_output_dir,
                seed=seed,
                epochs_override=epochs,
                device=device,
                csv_path=csv_path,
                external_test_sets={},  # 调参时不需要测试集
            )
        except Exception as e:
            logging.error(f"Training failed for P={p}, K={k}: {e}")
            continue
        
        # 加载训练好的模型
        model = EcClassifier(**cfg.MODEL_CONFIG).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        
        logging.info(f"Model loaded from {best_model_path}")
        
        # Cluster-aware 5-fold CV 评估
        cv_results = cluster_aware_cv_evaluate(
            model=model,
            dataset=eval_dataset,
            cluster_ids=cluster_ids,
            device=device,
            eval_config=cfg.EVAL_CONFIG,
            k=cv_k,
            tau=cv_tau,
            n_splits=n_splits,
            seed=seed,
            batch_size=p * k,
            chunk_size=1024,  # 分块大小，避免 OOM
        )
        
        # 记录结果
        result = {
            'P': p,
            'K': k,
            'batch_size': p * k,
            'ckpt_path': best_model_path,
            **cv_results
        }
        all_results.append(result)
        
        logging.info(f"\nP={p}, K={k} CV Results:")
        logging.info(f"  Mean F1: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
    
    # 保存所有结果到 CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_base, "pk_cv_results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"\nAll results saved to: {csv_path}")
    
    # 选择最佳 P/K
    if all_results:
        best_result = max(all_results, key=lambda x: x['mean_f1'])
        
        best_pk_info = {
            'best_P': best_result['P'],
            'best_K': best_result['K'],
            'mean_f1': best_result['mean_f1'],
            'std_f1': best_result['std_f1'],
            'mean_precision': best_result['mean_precision'],
            'mean_recall': best_result['mean_recall'],
            'ckpt_path': best_result['ckpt_path'],
            'seed': seed,
            'n_splits': n_splits,
            'cv_k': cv_k,
            'cv_tau': cv_tau,
            'epochs': epochs,
        }
        
        json_path = os.path.join(output_base, "best_pk.json")
        with open(json_path, 'w') as f:
            json.dump(best_pk_info, f, indent=2)
        
        logging.info("\n" + "="*60)
        logging.info(f"BEST P/K SELECTED:")
        logging.info(f"  P={best_pk_info['best_P']}, K={best_pk_info['best_K']}")
        logging.info(f"  Mean F1: {best_pk_info['mean_f1']:.4f} ± {best_pk_info['std_f1']:.4f}")
        logging.info(f"  Checkpoint: {best_pk_info['ckpt_path']}")
        logging.info(f"  Saved to: {json_path}")
        logging.info("="*60)
        
        return best_pk_info
    else:
        logging.error("No successful training runs. Cannot select best P/K.")
        return None


# ============================================================================
# 命令行接口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage A: Cluster-aware 5-fold CV for P/K selection"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(EMBEDDING_PATHS.keys()),
        help="Model name (esm, cpt, sub, cpt_sub)"
    )
    parser.add_argument(
        "--cluster_tsv", type=str, required=True,
        help="Path to mmseqs cluster TSV file (rep<TAB>member format)"
    )
    parser.add_argument(
        "--pk_list", type=str, nargs='+', default=['64x4', '32x8', '16x16'],
        help="List of P/K combinations (e.g., 64x4 32x8 16x16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Number of training epochs for each P/K (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=2025,
        help="Random seed (default: 2025)"
    )
    parser.add_argument(
        "--cv_k", type=int, default=3,
        help="k value for kNN in CV evaluation (default: 3)"
    )
    parser.add_argument(
        "--cv_tau", type=float, default=0.35,
        help="tau threshold for voting in CV evaluation (default: 0.35)"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    
    args = parser.parse_args()
    
    # 解析 P/K 列表
    pk_list = parse_pk_list(args.pk_list)
    
    # 运行调参
    run_pk_tuning(
        model_name=args.model,
        cluster_tsv_path=args.cluster_tsv,
        pk_list=pk_list,
        epochs=args.epochs,
        seed=args.seed,
        cv_k=args.cv_k,
        cv_tau=args.cv_tau,
        n_splits=args.n_splits,
    )


if __name__ == '__main__':
    main()