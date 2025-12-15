import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple

# 导入我们的小工具箱
from utils import save_detailed_predictions

# ==============================================================================
# --- Collate Function (保持不变) ---
# ==============================================================================
def evaluation_collate_fn(batch):
    valid = [b for b in batch if not (isinstance(b, dict) and b.get("skip", False))]
    if len(valid) == 0:
        return {"empty": True}
    embedding_tensors = torch.stack([item['embedding_1280'] for item in valid])
    label_lists = [item['labels'] for item in valid]
    seq_ids = [item['seq_id'] for item in valid]
    return {"embedding_1280": embedding_tensors, "labels": label_lists, "seq_ids": seq_ids, "empty": False}

# ==============================================================================
# --- Evaluator 类 (V4 修正版) ---
# ==============================================================================
class Evaluator:
    def __init__(self, model: nn.Module, gallery_loader: DataLoader, 
                 device: torch.device, eval_config: Dict):
        self.model = model
        self.device = device
        # self.config 仅用于获取 'weighted'
        self.config = eval_config 
        self.weighted = self.config.get('weighted', True) # 从 config 中获取 'weighted'
        
        logging.info("Building the evaluation gallery (reference set)...")
        
        self.gallery_embeddings, self.gallery_labels, self.gallery_ids = self._get_all_embeddings_and_labels(gallery_loader, "Building Gallery")
        
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Evaluation gallery is empty! Evaluation will not be possible.")
        else:
            logging.info(f"Gallery built successfully with {len(self.gallery_labels)} entries.")

    @torch.no_grad()
    def _get_all_embeddings_and_labels(self, dataloader: DataLoader, desc: str) -> Tuple[torch.Tensor, List[List[str]], List[str]]:
        """内部函数：从给定的 dataloader 提取所有128维嵌入、标签和ID"""
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

    def _find_best_f1_per_sample(self, true_set: set, counter: Dict[str, float]) -> Tuple[float, float, float, float]:
        """
        内部辅助函数：(V3 不变)
        对 *单个样本* 的原始分数(counter)进行搜索，找到最佳F1, 及其对应的 P, R, 和阈值。
        返回: (best_f1, best_precision, best_recall, best_threshold)
        """
        if not counter or not true_set:
            return (0.0, 0.0, 0.0, 0.0)

        thresholds_to_try = sorted(list(counter.values()), reverse=True)
        if not thresholds_to_try:
            return (0.0, 0.0, 0.0, 0.0)

        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = 0.0 
        all_ecs_in_counter = list(counter.keys())

        for thr in thresholds_to_try:
            pred_set = {ec for ec in all_ecs_in_counter if counter[ec] >= thr}
            
            if not pred_set:
                continue 

            tp = len(true_set & pred_set)
            prec = tp / len(pred_set)
            rec = tp / len(true_set)
            
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_precision = prec
                best_recall = rec
                best_threshold = thr
            elif f1 == best_f1:
                best_precision = prec
                best_recall = rec
                best_threshold = thr
        
        return best_f1, best_precision, best_recall, best_threshold

    # <<< V4 核心修改: k 和 vote_tau 作为参数传入 >>>
    def evaluate(self, query_loader: DataLoader, k: int, vote_tau: float, 
                 query_set_name: str = "Query Set", save_preds: bool = False, 
                 output_dir: str = ".") -> Dict[str, float]:
        """对给定的查询集 (Query Set) 执行评估"""
        
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Cannot evaluate because the gallery is empty.")
            return {}

        query_embeddings, query_labels, query_ids = self._get_all_embeddings_and_labels(query_loader, f"Processing {query_set_name}")

        if query_embeddings.numel() == 0:
            logging.warning(f"Query set '{query_set_name}' is empty after runtime skips. Returning 0.0 for all metrics.")
            return {}

        # k 和 vote_tau 来自函数参数
        # weighted 来自 self.config
        
        sims = torch.matmul(query_embeddings, self.gallery_embeddings.T)
        topk_sims, topk_idx = sims.topk(k, largest=True, dim=1)

        f1_scores_fixed_tau, precision_scores_fixed, recall_scores_fixed, auc_scores = [], [], [], []
        f1_max_scores = [] 
        precision_at_f1_max_scores = []
        recall_at_f1_max_scores = []
        best_threshold_scores = []
        prediction_records = []

        for i in range(len(query_labels)):
            true_set = set(query_labels[i])
            
            counter = defaultdict(float)
            weights_sum = 0.0
            for j, idx in enumerate(topk_idx[i].tolist()):
                w = float(topk_sims[i, j].item()) if self.weighted else 1.0
                weights_sum += w
                for ec in self.gallery_labels[idx]:
                    counter[ec] += w

            # --- 2. 使用 *传入的 vote_tau* 计算 F1/P/R ---
            thr_ref = weights_sum if self.weighted else k
            thr = vote_tau * thr_ref
            pred_set = {ec for ec, score in counter.items() if score >= thr}
            if not pred_set and counter:
                pred_set = {max(counter.items(), key=lambda x: x[1])[0]}

            if save_preds: 
                record = {
                    "Query_ID": query_ids[i],
                    "True_Labels": list(true_set),
                    "Predicted_Labels": list(pred_set) 
                }
                prediction_records.append(record)

            tp = len(true_set & pred_set)
            prec_fixed = tp / len(pred_set) if pred_set else 0.0
            rec_fixed = tp / len(true_set) if true_set else 0.0
            f1_fixed = 2 * prec_fixed * rec_fixed / (prec_fixed + rec_fixed) if (prec_fixed + rec_fixed) > 0 else 0.0
            
            f1_scores_fixed_tau.append(f1_fixed)
            precision_scores_fixed.append(prec_fixed) 
            recall_scores_fixed.append(rec_fixed)     

            # --- 3. (原逻辑) 计算 AUC (基于原始分数) ---
            if counter and true_set: 
                all_scored_ecs = list(counter.keys())
                y_true_auc = [1 if ec in true_set else 0 for ec in all_scored_ecs]
                y_score_auc = [counter[ec] for ec in all_scored_ecs]
                
                if len(set(y_true_auc)) > 1: 
                    try:
                        auc_scores.append(roc_auc_score(y_true_auc, y_score_auc))
                    except ValueError:
                        pass 
            
            # --- 4. (V3 逻辑) 计算 F1-max (用于分析) ---
            f1_max, p_max, r_max, tau_max = self._find_best_f1_per_sample(true_set, counter)
            
            f1_max_scores.append(f1_max)
            precision_at_f1_max_scores.append(p_max)
            recall_at_f1_max_scores.append(r_max)
            best_threshold_scores.append(tau_max)

        if save_preds:
            preds_save_path = os.path.join(output_dir, f"predictions_{query_set_name.replace(' ', '_').lower()}.csv")
            save_detailed_predictions(prediction_records, preds_save_path)

        # (V3 逻辑) 返回所有指标，用于调优和分析
        return {
            f'f1_at_fixed_tau_{vote_tau:.2f}': float(np.mean(f1_scores_fixed_tau)) if f1_scores_fixed_tau else 0.0,
            f'precision_at_fixed_tau_{vote_tau:.2f}': float(np.mean(precision_scores_fixed)) if precision_scores_fixed else 0.0,
            f'recall_at_fixed_tau_{vote_tau:.2f}': float(np.mean(recall_scores_fixed)) if recall_scores_fixed else 0.0,
            
            'auc': float(np.mean(auc_scores)) if auc_scores else 0.0,
            'f1_max': float(np.mean(f1_max_scores)) if f1_max_scores else 0.0,
            'precision_at_f1_max': float(np.mean(precision_at_f1_max_scores)) if precision_at_f1_max_scores else 0.0,
            'recall_at_f1_max': float(np.mean(recall_at_f1_max_scores)) if recall_at_f1_max_scores else 0.0,
            'avg_best_threshold': float(np.mean(best_threshold_scores)) if best_threshold_scores else 0.0
        }
