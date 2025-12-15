# evaluation.py
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
    valid = [b for b in batch if not (isinstance(b, dict) and b.get("skip", False))]
    if len(valid) == 0:
        return {"empty": True}
    embedding_tensors = torch.stack([item['embedding_1280'] for item in valid])
    label_lists = [item['labels'] for item in valid]
    seq_ids = [item['seq_id'] for item in valid]
    return {"embedding_1280": embedding_tensors, "labels": label_lists, "seq_ids": seq_ids, "empty": False}

class Evaluator:
    def __init__(self, model: nn.Module, gallery_loader: DataLoader, 
                 device: torch.device, eval_config: Dict):
        self.model = model
        self.device = device
        self.config = eval_config
        
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

    def evaluate(self, query_loader: DataLoader, k: int, vote_tau: float, 
                 query_set_name: str = "Query Set", save_preds: bool = False, 
                 output_dir: str = ".") -> Dict[str, float]:
        """对给定的查询集执行评估，使用指定的 k 和 vote_tau"""
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Cannot evaluate because the gallery is empty.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0}

        query_embeddings, query_labels, query_ids = self._get_all_embeddings_and_labels(query_loader, f"Processing {query_set_name}")

        if query_embeddings.numel() == 0:
            logging.warning(f"Query set '{query_set_name}' is empty after runtime skips. Returning 0.0 for all metrics.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0}

        weighted = self.config['weighted']

        sims = torch.matmul(query_embeddings, self.gallery_embeddings.T)
        topk_sims, topk_idx = sims.topk(k, largest=True, dim=1)

        f1_scores, precision_scores, recall_scores, auc_scores = [], [], [], []
        prediction_records = []

        for i in range(len(query_labels)):
            true_set = set(query_labels[i])
            
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

            tp = len(true_set & pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec = tp / len(true_set) if true_set else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            f1_scores.append(f1)
            precision_scores.append(prec)
            recall_scores.append(rec)

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
            preds_save_path = os.path.join(output_dir, f"predictions_{query_set_name.replace(' ', '_').lower()}.csv")
            save_detailed_predictions(prediction_records, preds_save_path)

        return {
            'f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
            'precision': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'recall': float(np.mean(recall_scores)) if recall_scores else 0.0,
            'auc': float(np.mean(auc_scores)) if auc_scores else 0.0
        }