#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tune_ktau_cv.py - Stage B: Cluster-aware 5-fold CV for k/tau selection

在 Stage A 选出的 best (P,K) 对应的模型上，做 k/tau 网格搜索。
不需要重新训练，只需要反复调用 evaluator.evaluate()。

用法示例:
    python tune_ktau_cv.py --model esm --cluster_tsv /path/to/cluster.tsv \
        --ckpt_path experiments/tune/esm/P32_K8/best_model.pth \
        --k_list 3 5 7 10 --tau_list 0.2 0.25 0.3 0.35 0.4 0.45 0.5

    # 或者从 best_pk.json 自动读取 ckpt_path:
    python tune_ktau_cv.py --model esm --cluster_tsv /path/to/cluster.tsv --auto

输出:
    experiments/tune/<model>/ktau_cv_results.csv
    experiments/tune/<model>/best_ktau.json
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
from main import set_seeds
from tune_pk_cv import (
    EMBEDDING_PATHS,
    parse_cluster_tsv,
    get_cluster_ids_for_dataset,
)


# ============================================================================
# k/tau 网格搜索 (Cluster-aware CV) - 内存优化版
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
    chunk_size: int = 1024,
) -> Dict[str, float]:
    """
    分块计算 kNN 评估，避免 OOM。
    """
    Q, D = query_emb.shape
    G = gallery_emb.shape[0]
    
    gallery_gpu = gallery_emb.to(device)
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for start_idx in range(0, Q, chunk_size):
        end_idx = min(start_idx + chunk_size, Q)
        
        query_chunk = query_emb[start_idx:end_idx].to(device)
        chunk_labels = query_labels[start_idx:end_idx]
        
        sims = torch.matmul(query_chunk.float(), gallery_gpu.T.float())
        
        k_eff = min(k, G)
        topk_sims, topk_idx = sims.topk(k_eff, largest=True, dim=1)
        
        topk_sims = topk_sims.cpu()
        topk_idx = topk_idx.cpu()
        
        for i in range(len(chunk_labels)):
            true_set = set(chunk_labels[i])
            
            counter = defaultdict(float)
            if weighted:
                w_vec = torch.softmax(topk_sims[i].float(), dim=0)
                for jpos, idx in enumerate(topk_idx[i].tolist()):
                    w = float(w_vec[jpos].item())
                    for ec in gallery_labels[idx]:
                        counter[ec] += w
                thr_ref = 1.0
            else:
                for idx in topk_idx[i].tolist():
                    for ec in gallery_labels[idx]:
                        counter[ec] += 1.0
                thr_ref = float(k_eff)
            
            thr = tau * thr_ref
            pred_set = {ec for ec, score in counter.items() if score >= thr}
            
            if not pred_set and counter:
                pred_set = {max(counter.items(), key=lambda x: x[1])[0]}
            
            tp = len(true_set & pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec = tp / len(true_set) if true_set else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            f1_scores.append(f1)
            precision_scores.append(prec)
            recall_scores.append(rec)
        
        del query_chunk, sims, topk_sims, topk_idx
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    del gallery_gpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
        'precision': float(np.mean(precision_scores)) if precision_scores else 0.0,
        'recall': float(np.mean(recall_scores)) if recall_scores else 0.0,
        'auc': 0.0,
        'auc_macro': 0.0,
    }


def cluster_aware_cv_grid_search(
    model: torch.nn.Module,
    dataset: EvaluationDataset,
    cluster_ids: List[str],
    device: torch.device,
    eval_config: Dict,
    k_list: List[int],
    tau_list: List[float],
    n_splits: int = 5,
    seed: int = 42,
    batch_size: int = 256,
    chunk_size: int = 1024,
) -> Tuple[List[Dict], Dict]:
    """
    使用 GroupKFold 进行 cluster-aware 5-fold CV k/tau 网格搜索（内存优化版）。
    
    优化策略：
    - 预先计算并缓存所有折的 embeddings（存在 CPU 上）
    - 对不同的 k/tau，直接在缓存的 embeddings 上计算
    - 分块计算相似度矩阵避免 OOM
    
    Args:
        model: 训练好的模型
        dataset: EvaluationDataset
        cluster_ids: 每个样本对应的 cluster_id
        device: 计算设备
        eval_config: 评估配置
        k_list: k 值候选列表
        tau_list: tau 值候选列表
        n_splits: CV 折数
        seed: 随机种子
        batch_size: batch size for encoding
        chunk_size: chunk size for similarity computation
        
    Returns:
        all_results: 所有 (k, tau) 组合的结果列表
        best_result: 最佳结果
    """
    set_seeds(seed)
    
    # 将 cluster_id 转换为整数编码
    unique_clusters = sorted(set(cluster_ids))
    cluster_to_int = {c: i for i, c in enumerate(unique_clusters)}
    groups = np.array([cluster_to_int[c] for c in cluster_ids])
    
    logging.info(f"CV setup: {len(dataset)} samples, {len(unique_clusters)} clusters, {n_splits} folds")
    logging.info(f"Grid search: {len(k_list)} k values × {len(tau_list)} tau values = {len(k_list) * len(tau_list)} combinations")
    logging.info(f"Memory optimization: chunk_size={chunk_size}")
    
    # GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(dataset))
    weighted = eval_config.get('weighted', True)
    
    # 预先计算并缓存所有折的 embeddings
    fold_data = []
    
    logging.info("Pre-computing embeddings for all folds (stored on CPU)...")
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        gallery_subset = Subset(dataset, train_idx.tolist())
        query_subset = Subset(dataset, val_idx.tolist())
        
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
        
        # 提取并缓存 embeddings（存在 CPU 上）
        gallery_emb, gallery_labels, _ = _get_embeddings_from_loader(
            model, gallery_loader, device, normalize=True, desc=f"Fold {fold_idx+1} Gallery"
        )
        query_emb, query_labels, _ = _get_embeddings_from_loader(
            model, query_loader, device, normalize=True, desc=f"Fold {fold_idx+1} Query"
        )
        
        fold_data.append({
            'fold_idx': fold_idx,
            'gallery_emb': gallery_emb,  # CPU tensor
            'gallery_labels': gallery_labels,
            'query_emb': query_emb,      # CPU tensor
            'query_labels': query_labels,
        })
        
        logging.info(f"  Fold {fold_idx + 1}: gallery={len(train_idx)}, query={len(val_idx)}")
    
    # 清理显存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 网格搜索
    all_results = []
    total_combinations = len(k_list) * len(tau_list)
    
    progress_bar = tqdm(
        total=total_combinations,
        desc="Grid searching k/tau"
    )
    
    for k in k_list:
        for tau in tau_list:
            fold_metrics = {
                'f1': [], 'precision': [], 'recall': [], 'auc': [], 'auc_macro': []
            }
            
            for fold_info in fold_data:
                # 使用缓存的 embeddings 进行评估
                metrics = _chunked_knn_evaluate(
                    query_emb=fold_info['query_emb'],
                    query_labels=fold_info['query_labels'],
                    gallery_emb=fold_info['gallery_emb'],
                    gallery_labels=fold_info['gallery_labels'],
                    device=device,
                    k=k,
                    tau=tau,
                    weighted=weighted,
                    chunk_size=chunk_size,
                )
                
                for metric_name in fold_metrics.keys():
                    if metric_name in metrics:
                        fold_metrics[metric_name].append(metrics[metric_name])
            
            # 计算 mean ± std
            result = {
                'k': k,
                'tau': tau,
            }
            for metric_name, values in fold_metrics.items():
                if values:
                    result[f'mean_{metric_name}'] = float(np.mean(values))
                    result[f'std_{metric_name}'] = float(np.std(values))
            
            all_results.append(result)
            
            progress_bar.set_postfix(
                k=k, tau=f"{tau:.2f}",
                f1=f"{result['mean_f1']:.4f}"
            )
            progress_bar.update(1)
    
    progress_bar.close()
    
    # 选择最佳结果
    best_result = max(all_results, key=lambda x: x['mean_f1'])
    
    logging.info("\n" + "="*60)
    logging.info("Grid Search Summary:")
    logging.info(f"  Total combinations evaluated: {len(all_results)}")
    logging.info(f"  Best k: {best_result['k']}")
    logging.info(f"  Best tau: {best_result['tau']:.4f}")
    logging.info(f"  Best Mean F1: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")
    logging.info(f"  Best Mean P:  {best_result['mean_precision']:.4f}")
    logging.info(f"  Best Mean R:  {best_result['mean_recall']:.4f}")
    logging.info("="*60)
    
    return all_results, best_result


# ============================================================================
# k/tau 调参主函数
# ============================================================================
def run_ktau_tuning(
    model_name: str,
    cluster_tsv_path: str,
    ckpt_path: str,
    k_list: List[int],
    tau_list: List[float],
    seed: int = 42,
    n_splits: int = 5,
    batch_size: int = 256,
):
    """
    k/tau 调参主函数。
    
    加载已训练好的模型，在训练集上做 cluster-aware 5-fold CV 网格搜索。
    """
    # 获取 embedding 路径
    if model_name not in EMBEDDING_PATHS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(EMBEDDING_PATHS.keys())}")
    embedding_dir = EMBEDDING_PATHS[model_name]
    
    # 输出目录
    output_base = os.path.join("experiments", "tune", model_name)
    os.makedirs(output_base, exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(output_base, f"ktau_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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
    logging.info(f"k/tau Tuning for model: {model_name}")
    logging.info(f"Embedding dir: {embedding_dir}")
    logging.info(f"Cluster TSV: {cluster_tsv_path}")
    logging.info(f"Checkpoint: {ckpt_path}")
    logging.info(f"k candidates: {k_list}")
    logging.info(f"tau candidates: {tau_list}")
    logging.info(f"CV settings: n_splits={n_splits}, seed={seed}")
    logging.info("="*60)
    
    # 检查 checkpoint 是否存在
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # 解析聚类文件
    member_to_cluster = parse_cluster_tsv(cluster_tsv_path)
    
    # 加载数据集
    csv_path = cfg.DATA_PATHS["csv_path"]
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(csv_path)
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    # 增强标签映射
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
    
    # 获取 cluster_ids
    seq_ids = [eval_dataset.ids[i] for i in range(len(eval_dataset))]
    cluster_ids, n_missing = get_cluster_ids_for_dataset(seq_ids, member_to_cluster)
    
    logging.info(f"Prepared CV dataset: {len(eval_dataset)} samples")
    logging.info(f"Cluster mapping: {len(cluster_ids)} IDs, {n_missing} missing")
    
    # 加载模型
    device = cfg.DEVICE
    model = EcClassifier(**cfg.MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    logging.info(f"Model loaded from {ckpt_path}")
    
    # 网格搜索
    all_results, best_result = cluster_aware_cv_grid_search(
        model=model,
        dataset=eval_dataset,
        cluster_ids=cluster_ids,
        device=device,
        eval_config=cfg.EVAL_CONFIG,
        k_list=k_list,
        tau_list=tau_list,
        n_splits=n_splits,
        seed=seed,
        batch_size=batch_size,
        chunk_size=1024,  # 分块大小，避免 OOM
    )
    
    # 保存所有结果到 CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_base, "ktau_cv_results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"All results saved to: {csv_path}")
    
    # 保存最佳结果
    best_ktau_info = {
        'best_k': best_result['k'],
        'best_tau': best_result['tau'],
        'mean_f1': best_result['mean_f1'],
        'std_f1': best_result['std_f1'],
        'mean_precision': best_result['mean_precision'],
        'mean_recall': best_result['mean_recall'],
        'ckpt_path': ckpt_path,
        'seed': seed,
        'n_splits': n_splits,
        'k_search_space': k_list,
        'tau_search_space': tau_list,
    }
    
    json_path = os.path.join(output_base, "best_ktau.json")
    with open(json_path, 'w') as f:
        json.dump(best_ktau_info, f, indent=2)
    
    logging.info(f"\nBest k/tau saved to: {json_path}")
    
    # 打印热力图式的结果摘要
    logging.info("\n" + "="*60)
    logging.info("F1 Score Matrix (k × tau):")
    logging.info("="*60)
    
    # 构建热力图数据
    pivot_df = results_df.pivot(index='k', columns='tau', values='mean_f1')
    logging.info("\n" + pivot_df.to_string())
    
    return best_ktau_info


# ============================================================================
# 命令行接口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage B: Cluster-aware 5-fold CV for k/tau selection"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(EMBEDDING_PATHS.keys()),
        help="Model name (esm, cpt, sub, cpt_sub)"
    )
    parser.add_argument(
        "--cluster_tsv", type=str, required=True,
        help="Path to mmseqs cluster TSV file"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to model checkpoint. If not provided and --auto is set, read from best_pk.json"
    )
    parser.add_argument(
        "--auto", action='store_true',
        help="Auto-load checkpoint path from experiments/tune/<model>/best_pk.json"
    )
    parser.add_argument(
        "--k_list", type=int, nargs='+', default=[3, 5, 7, 10],
        help="List of k values to search (default: 3 5 7 10)"
    )
    parser.add_argument(
        "--tau_list", type=float, nargs='+', 
        default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        help="List of tau values to search"
    )
    parser.add_argument(
        "--seed", type=int, default=2025,
        help="Random seed (default: 2025)"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for evaluation (default: 256)"
    )
    
    args = parser.parse_args()
    
    # 确定 checkpoint 路径
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        if args.auto:
            # 从 best_pk.json 读取
            best_pk_path = os.path.join("experiments", "tune", args.model, "best_pk.json")
            if not os.path.exists(best_pk_path):
                raise FileNotFoundError(
                    f"best_pk.json not found at {best_pk_path}. "
                    "Please run tune_pk_cv.py first or specify --ckpt_path"
                )
            with open(best_pk_path, 'r') as f:
                best_pk_info = json.load(f)
            ckpt_path = best_pk_info['ckpt_path']
            print(f"Auto-loaded checkpoint from best_pk.json: {ckpt_path}")
        else:
            raise ValueError("Please specify --ckpt_path or use --auto flag")
    
    # 运行调参
    run_ktau_tuning(
        model_name=args.model,
        cluster_tsv_path=args.cluster_tsv,
        ckpt_path=ckpt_path,
        k_list=args.k_list,
        tau_list=args.tau_list,
        seed=args.seed,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()