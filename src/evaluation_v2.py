# evaluation_v2.py
"""
增强版评估模块
- 支持全局AUPRC计算和曲线绘制
- 保留原有的kNN投票评估逻辑
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    auc
)
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

from utils import save_detailed_predictions


def evaluation_collate_fn(batch):
    valid = [b for b in batch if not (isinstance(b, dict) and b.get("skip", False))]
    if len(valid) == 0:
        return {"empty": True}
    embedding_tensors = torch.stack([item['embedding_1280'] for item in valid])
    label_lists = [item['labels'] for item in valid]
    seq_ids = [item['seq_id'] for item in valid]
    return {"embedding_1280": embedding_tensors, "labels": label_lists, "seq_ids": seq_ids, "empty": False}


class Evaluator:
    """
    评估器类
    - 构建gallery（参考库）
    - 使用kNN加权投票进行EC号预测
    - 支持AUPRC曲线绘制
    """
    
    def __init__(self, model: nn.Module, gallery_loader: DataLoader, 
                 device: torch.device, eval_config: Dict):
        self.model = model
        self.device = device
        self.config = eval_config
        
        logging.info("Building the evaluation gallery (reference set)...")
        
        self.gallery_embeddings, self.gallery_labels, self.gallery_ids = \
            self._get_all_embeddings_and_labels(gallery_loader, "Building Gallery")
        
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Evaluation gallery is empty! Evaluation will not be possible.")
        else:
            logging.info(f"Gallery built successfully with {len(self.gallery_labels)} entries.")

    @torch.no_grad()
    def _get_all_embeddings_and_labels(self, dataloader: DataLoader, desc: str) -> Tuple[torch.Tensor, List[List[str]], List[str]]:
        """从给定的 dataloader 提取所有128维嵌入、标签和ID"""
        self.model.eval()
        all_embeddings_128, all_labels, all_ids = [], [], []
        
        for batch in tqdm(dataloader, desc=desc):
            if isinstance(batch, dict) and batch.get("empty", False):
                continue
            embeddings_1280 = batch['embedding_1280'].to(self.device, non_blocking=True)
            with autocast(enabled=True):
                embeddings_128 = self.model(embeddings_1280)
            all_embeddings_128.append(embeddings_128.cpu())
            all_labels.extend(batch['labels'])
            all_ids.extend(batch['seq_ids'])
            
        if not all_labels:
            output_dim = self.model.fc3.out_features if hasattr(self.model, 'fc3') else 128
            return torch.empty((0, output_dim)), [], []
        
        return torch.cat(all_embeddings_128, dim=0), all_labels, all_ids

    def evaluate(self, query_loader: DataLoader, k: int, vote_tau: float, 
                 query_set_name: str = "Query Set", save_preds: bool = False, 
                 output_dir: str = ".",
                 plot_auprc: bool = False) -> Dict[str, float]:
        """
        对给定的查询集执行评估
        
        Args:
            query_loader: 查询集数据加载器
            k: kNN的k值
            vote_tau: 投票阈值
            query_set_name: 查询集名称（用于日志和文件命名）
            save_preds: 是否保存预测结果
            output_dir: 输出目录
            plot_auprc: 是否绘制AUPRC曲线
            
        Returns:
            Dict[str, float]: 包含各项指标的字典
        """
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Cannot evaluate because the gallery is empty.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'auprc': 0.0}

        query_embeddings, query_labels, query_ids = \
            self._get_all_embeddings_and_labels(query_loader, f"Processing {query_set_name}")

        if query_embeddings.numel() == 0:
            logging.warning(f"Query set '{query_set_name}' is empty. Returning 0.0 for all metrics.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'auprc': 0.0}

        weighted = self.config.get('weighted', True)

        # 计算相似度矩阵并获取top-k
        sims = torch.matmul(query_embeddings, self.gallery_embeddings.T)
        topk_sims, topk_idx = sims.topk(k, largest=True, dim=1)

        # 用于计算传统指标
        f1_scores, precision_scores, recall_scores, auc_scores = [], [], [], []
        prediction_records = []
        
        # 用于全局AUPRC计算：收集所有 (query, EC) 对的分数和标签
        global_y_true = []  # 1 if EC is true label, 0 otherwise
        global_y_score = [] # normalized score for this EC

        for i in range(len(query_labels)):
            true_set = set(query_labels[i])
            
            # 统计投票
            counter = defaultdict(float)
            weights_sum = 0.0
            for j, idx in enumerate(topk_idx[i].tolist()):
                w = float(topk_sims[i, j].item()) if weighted else 1.0
                weights_sum += w
                for ec in self.gallery_labels[idx]:
                    counter[ec] += w

            # 归一化分数并收集全局AUPRC数据
            if counter and weights_sum > 0:
                for ec, score in counter.items():
                    normalized_score = score / weights_sum
                    global_y_score.append(normalized_score)
                    global_y_true.append(1 if ec in true_set else 0)

            # 投票阈值预测
            thr_ref = weights_sum if weighted else k
            thr = vote_tau * thr_ref
            pred_set = {ec for ec, score in counter.items() if score >= thr}
            if not pred_set and counter:
                pred_set = {max(counter.items(), key=lambda x: x[1])[0]}

            if save_preds:
                # 按分数排序的预测结果
                sorted_preds = sorted(counter.items(), key=lambda x: -x[1])
                record = {
                    "Query_ID": query_ids[i],
                    "True_Labels": list(true_set),
                    "Predicted_Labels": list(pred_set),
                    "All_Scores": {ec: f"{s/weights_sum:.4f}" for ec, s in sorted_preds[:10]}  # top10
                }
                prediction_records.append(record)

            # 计算传统指标
            tp = len(true_set & pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec = tp / len(true_set) if true_set else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            f1_scores.append(f1)
            precision_scores.append(prec)
            recall_scores.append(rec)

            # Sample-wise AUC（保留原逻辑）
            if counter:
                all_scored_ecs = list(counter.keys())
                y_true_auc = [1 if ec in true_set else 0 for ec in all_scored_ecs]
                y_score_auc = [counter[ec] for ec in all_scored_ecs]
                
                if len(set(y_true_auc)) > 1:
                    try:
                        auc_scores.append(roc_auc_score(y_true_auc, y_score_auc))
                    except ValueError:
                        pass

        # 保存预测结果
        if save_preds:
            preds_save_path = os.path.join(output_dir, f"predictions_{query_set_name.replace(' ', '_').lower()}.csv")
            save_detailed_predictions(prediction_records, preds_save_path)

        # 计算全局AUPRC
        global_auprc = 0.0
        if global_y_true and len(set(global_y_true)) > 1:
            global_y_true = np.array(global_y_true)
            global_y_score = np.array(global_y_score)
            global_auprc = average_precision_score(global_y_true, global_y_score)
            
            # 绘制AUPRC曲线
            if plot_auprc:
                self._plot_pr_curve(
                    global_y_true, 
                    global_y_score, 
                    global_auprc,
                    query_set_name, 
                    output_dir
                )

        return {
            'f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
            'precision': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'recall': float(np.mean(recall_scores)) if recall_scores else 0.0,
            'auc': float(np.mean(auc_scores)) if auc_scores else 0.0,
            'auprc': float(global_auprc)
        }

    def _plot_pr_curve(self, y_true: np.ndarray, y_score: np.ndarray, 
                       auprc: float, set_name: str, output_dir: str):
        """绘制并保存Precision-Recall曲线"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        # 计算baseline（随机分类器的期望precision）
        baseline = np.sum(y_true) / len(y_true)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 主曲线
        ax.plot(recall, precision, 'b-', linewidth=2, 
                label=f'PR Curve (AUPRC = {auprc:.4f})')
        
        # Baseline
        ax.axhline(y=baseline, color='r', linestyle='--', linewidth=1.5,
                   label=f'Random Baseline = {baseline:.4f}')
        
        # 填充曲线下方区域
        ax.fill_between(recall, precision, alpha=0.2)
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve: {set_name}', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        n_positive = int(np.sum(y_true))
        n_total = len(y_true)
        info_text = f'Positive: {n_positive:,} / {n_total:,} ({100*baseline:.2f}%)'
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"auprc_curve_{set_name.replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"AUPRC curve saved to: {save_path}")
        
        return save_path


def plot_multiple_pr_curves(results_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]], 
                            output_path: str, title: str = "Precision-Recall Curves Comparison"):
    """
    在同一张图上绘制多个测试集的PR曲线
    
    Args:
        results_dict: {set_name: (y_true, y_score, auprc)}
        output_path: 保存路径
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for idx, (set_name, (y_true, y_score, auprc)) in enumerate(results_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, color=colors[idx], linewidth=2,
                label=f'{set_name} (AUPRC = {auprc:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Combined PR curves saved to: {output_path}")