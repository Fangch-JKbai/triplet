# evaluation_v3.py (综合评估版本)
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

from utils import save_detailed_predictions

def evaluation_collate_fn(batch):
    """Collate function for evaluation dataloader"""
    valid = [b for b in batch if not (isinstance(b, dict) and b.get("skip", False))]
    if len(valid) == 0:
        return {"empty": True}
    embedding_tensors = torch.stack([item['embedding_1280'] for item in valid])
    label_lists = [item['labels'] for item in valid]
    seq_ids = [item['seq_id'] for item in valid]
    return {"embedding_1280": embedding_tensors, "labels": label_lists, "seq_ids": seq_ids, "empty": False}


class ComprehensiveEvaluator:
    """
    综合评估器：
    1. 传统的下游任务评估 (k-NN + voting)
    2. Embedding内在质量评估
    """
    def __init__(self, model: nn.Module, gallery_loader: DataLoader, 
                 device: torch.device, eval_config: Dict):
        self.model = model
        self.device = device
        self.config = eval_config
        
        logging.info("Building the evaluation gallery (reference set)...")
        
        # 构建gallery
        self.gallery_embeddings, self.gallery_labels, self.gallery_ids = \
            self._get_all_embeddings_and_labels(gallery_loader, "Building Gallery")
        
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Evaluation gallery is empty!")
        else:
            logging.info(f"Gallery built with {len(self.gallery_labels)} entries.")
    
    @torch.no_grad()
    def _get_all_embeddings_and_labels(self, dataloader: DataLoader, desc: str) -> Tuple[torch.Tensor, List[List[str]], List[str]]:
        """从dataloader提取所有128维嵌入、标签和ID"""
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
    
    def evaluate_downstream_task(self, query_loader: DataLoader, k: int, vote_tau: float,
                                 query_set_name: str = "Query Set", 
                                 save_preds: bool = False, 
                                 output_dir: str = ".") -> Dict[str, float]:
        """
        执行传统的下游任务评估 (k-NN + voting)
        """
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Cannot evaluate: gallery is empty.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0}
        
        query_embeddings, query_labels, query_ids = \
            self._get_all_embeddings_and_labels(query_loader, f"Processing {query_set_name}")
        
        if query_embeddings.numel() == 0:
            logging.warning(f"Query set '{query_set_name}' is empty.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0}
        
        weighted = self.config.get('weighted', True)
        
        # 计算相似度
        sims = torch.matmul(query_embeddings, self.gallery_embeddings.T)
        topk_sims, topk_idx = sims.topk(k, largest=True, dim=1)
        
        f1_scores, precision_scores, recall_scores, auc_scores = [], [], [], []
        prediction_records = []
        
        for i in range(len(query_labels)):
            true_set = set(query_labels[i])
            
            # 投票
            counter = defaultdict(float)
            weights_sum = 0.0
            for j, idx in enumerate(topk_idx[i].tolist()):
                w = float(topk_sims[i, j].item()) if weighted else 1.0
                weights_sum += w
                for ec in self.gallery_labels[idx]:
                    counter[ec] += w
            
            thr_ref = weights_sum if weighted else k
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
            
            # 计算指标
            tp = len(true_set & pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec = tp / len(true_set) if true_set else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            f1_scores.append(f1)
            precision_scores.append(prec)
            recall_scores.append(rec)
            
            # AUC
            if counter:
                all_scored_ecs = list(counter.keys())
                y_true_auc = [1 if ec in true_set else 0 for ec in all_scored_ecs]
                y_score_auc = [counter[ec] for ec in all_scored_ecs]
                
                if len(set(y_true_auc)) > 1:
                    try:
                        auc_scores.append(roc_auc_score(y_true_auc, y_score_auc))
                    except ValueError:
                        pass
        
        if save_preds:
            preds_save_path = os.path.join(output_dir, 
                                          f"predictions_{query_set_name.replace(' ', '_').lower()}.csv")
            save_detailed_predictions(prediction_records, preds_save_path)
        
        return {
            f'f1_at_fixed_tau_{vote_tau:.2f}': float(np.mean(f1_scores)) if f1_scores else 0.0,
            f'precision_at_fixed_tau_{vote_tau:.2f}': float(np.mean(precision_scores)) if precision_scores else 0.0,
            f'recall_at_fixed_tau_{vote_tau:.2f}': float(np.mean(recall_scores)) if recall_scores else 0.0,
            'auc': float(np.mean(auc_scores)) if auc_scores else 0.0
        }
    
    def tune_hyperparameters(self, val_loader: DataLoader, 
                            k_search: List[int], 
                            tau_search: np.ndarray) -> Tuple[int, float, Dict]:
        """
        在验证集上搜索最佳k和tau
        
        Returns:
            (best_k, best_tau, tuning_heatmap_data)
        """
        logging.info("="*50)
        logging.info("Starting Hyperparameter Tuning")
        logging.info("="*50)
        
        best_f1 = -1.0
        best_k = k_search[0]
        best_tau = tau_search[0]
        
        # 用于可视化的热图数据
        heatmap_data = np.zeros((len(k_search), len(tau_search)))
        
        tuning_pbar = tqdm(total=len(k_search) * len(tau_search), 
                          desc="Tuning k/tau")
        
        for i, k in enumerate(k_search):
            for j, tau in enumerate(tau_search):
                metrics = self.evaluate_downstream_task(
                    val_loader, k=k, vote_tau=tau,
                    query_set_name="Tuning", save_preds=False
                )
                
                key = f'f1_at_fixed_tau_{tau:.2f}'
                current_f1 = metrics.get(key, 0.0)
                heatmap_data[i, j] = current_f1
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_k = k
                    best_tau = tau
                
                tuning_pbar.set_postfix(k=k, tau=f"{tau:.2f}", 
                                       best_f1=f"{best_f1:.4f}")
                tuning_pbar.update(1)
        
        tuning_pbar.close()
        logging.info(f"Best hyperparameters: k={best_k}, tau={best_tau:.4f}, F1={best_f1:.4f}")
        
        tuning_results = {
            'k_values': k_search,
            'tau_values': tau_search.tolist(),
            'heatmap': heatmap_data.tolist(),
            'best_k': best_k,
            'best_tau': best_tau,
            'best_f1': best_f1
        }
        
        return best_k, best_tau, tuning_results