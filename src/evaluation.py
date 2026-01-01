import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Any

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
    Evaluator for k-NN based retrieval/classification.
    
    Optimized to support:
    1. Mixed Precision (AMP)
    2. Efficient Matrix Multiplication (GPU/CPU handling)
    3. Returning raw scores for PR Curve plotting (avoids double inference)
    """

    def __init__(self, model: nn.Module, gallery_loader: DataLoader,
                 device: torch.device, eval_config: Dict):
        self.model = model
        self.device = device
        self.config = eval_config

        # Defaults
        self.config.setdefault("weighted", False)
        self.config.setdefault("weighting_mode", "softmax")  # "softmax" | "clamp" | "raw"
        self.config.setdefault("normalize_embeddings", True)
        self.config.setdefault("keep_gallery_on_device", True)
        self.config.setdefault("sim_dtype_fp32", True)

        logging.info("Building the evaluation gallery (reference set)...")
        gallery_emb, gallery_labels, gallery_ids = self._get_all_embeddings_and_labels(
            gallery_loader, "Building Gallery"
        )

        if gallery_emb.numel() == 0:
            logging.warning("Evaluation gallery is empty! Evaluation will not be possible.")
            self.gallery_embeddings = gallery_emb
            self.gallery_labels = gallery_labels
            self.gallery_ids = gallery_ids
            self.label_to_idx = {}
            self.idx_to_label = []
            return

        # Move gallery to desired device
        if self.config["keep_gallery_on_device"]:
            self.gallery_embeddings = gallery_emb.to(self.device, non_blocking=True)
        else:
            self.gallery_embeddings = gallery_emb.cpu()

        self.gallery_labels = gallery_labels
        self.gallery_ids = gallery_ids

        # Build label vocabulary from gallery
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: List[str] = []
        self._ensure_label_vocab_from_labels(self.gallery_labels)

        logging.info(f"Gallery built successfully with {len(self.gallery_labels)} entries.")
        logging.info(f"Label vocabulary size (from gallery): {len(self.idx_to_label)}")

    @torch.no_grad()
    def _get_all_embeddings_and_labels(
        self, dataloader: DataLoader, desc: str
    ) -> Tuple[torch.Tensor, List[List[str]], List[str]]:
        self.model.eval()
        all_embeddings_128: List[torch.Tensor] = []
        all_labels: List[List[str]] = []
        all_ids: List[str] = []

        use_amp = (self.device.type == "cuda")

        for batch in tqdm(dataloader, desc=desc):
            if isinstance(batch, dict) and batch.get("empty", False):
                continue

            embeddings_1280 = batch['embedding_1280'].to(self.device, non_blocking=True)

            with autocast(enabled=use_amp):
                embeddings_128 = self.model(embeddings_1280)

            if self.config.get("normalize_embeddings", True):
                embeddings_128 = F.normalize(embeddings_128, dim=1)

            all_embeddings_128.append(embeddings_128.detach().cpu())
            all_labels.extend(batch['labels'])
            all_ids.extend(batch['seq_ids'])

        if not all_labels:
            # Fallback for empty shape
            output_dim = 128
            if hasattr(self.model, "fc3") and hasattr(self.model.fc3, "out_features"):
                output_dim = int(self.model.fc3.out_features)
            return torch.empty((0, output_dim)), [], []

        return torch.cat(all_embeddings_128, dim=0), all_labels, all_ids

    def _ensure_label_vocab_from_labels(self, label_lists: List[List[str]]) -> None:
        for labs in label_lists:
            for lb in labs:
                if lb not in self.label_to_idx:
                    self.label_to_idx[lb] = len(self.idx_to_label)
                    self.idx_to_label.append(lb)

    def _maybe_move_query_embeddings(self, query_embeddings_cpu: torch.Tensor) -> torch.Tensor:
        if self.config["keep_gallery_on_device"]:
            return query_embeddings_cpu.to(self.device, non_blocking=True)
        return query_embeddings_cpu.cpu()

    def _compute_multilabel_auc(
        self,
        y_true: np.ndarray,     # (N, L) binary
        y_score: np.ndarray     # (N, L) float
    ) -> Tuple[float, float]:
        micro_auc = 0.0
        macro_auc = 0.0

        # Micro
        y_true_flat = y_true.reshape(-1)
        if len(np.unique(y_true_flat)) > 1:
            try:
                micro_auc = float(roc_auc_score(y_true, y_score, average="micro"))
            except ValueError:
                micro_auc = 0.0

        # Macro
        per_label_aucs = []
        L = y_true.shape[1]
        for j in range(L):
            col = y_true[:, j]
            if len(np.unique(col)) <= 1:
                continue
            try:
                per_label_aucs.append(float(roc_auc_score(col, y_score[:, j])))
            except ValueError:
                continue

        if per_label_aucs:
            macro_auc = float(np.mean(per_label_aucs))

        return micro_auc, macro_auc

    @torch.no_grad()
    def evaluate(
        self,
        query_loader: DataLoader,
        k: int,
        vote_tau: float,
        query_set_name: str = "Query Set",
        save_preds: bool = False,
        output_dir: str = ".",
        plot_auprc: bool = False,         # Compatibility arg (unused logic inside, handled by return_raw_scores)
        return_raw_scores: bool = False   # <--- 新增核心参数
    ) -> Dict[str, Any]:
        
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Cannot evaluate because the gallery is empty.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'auc_macro': 0.0}

        query_emb_cpu, query_labels, query_ids = self._get_all_embeddings_and_labels(
            query_loader, f"Processing {query_set_name}"
        )

        if query_emb_cpu.numel() == 0:
            logging.warning(f"Query set '{query_set_name}' is empty. Returning 0.0.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'auc_macro': 0.0}

        self._ensure_label_vocab_from_labels(query_labels)
        L = len(self.idx_to_label)

        # 1. Calculate Similarity
        query_embeddings = self._maybe_move_query_embeddings(query_emb_cpu)
        gallery_embeddings = self.gallery_embeddings
        if not self.config["keep_gallery_on_device"]:
            gallery_embeddings = gallery_embeddings.cpu()

        if self.config.get("sim_dtype_fp32", True):
            sims = torch.matmul(query_embeddings.float(), gallery_embeddings.T.float())
        else:
            sims = torch.matmul(query_embeddings, gallery_embeddings.T)

        k_eff = int(min(k, sims.shape[1]))
        if k_eff <= 0:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'auc_macro': 0.0}

        topk_sims, topk_idx = sims.topk(k_eff, largest=True, dim=1)

        # 2. Setup Voting Config
        weighted = bool(self.config["weighted"])
        weighting_mode = str(self.config.get("weighting_mode", "softmax")).lower()

        f1_scores, precision_scores, recall_scores = [], [], []
        prediction_records = []

        # Standard AUC matrices
        y_true_mat = np.zeros((len(query_labels), L), dtype=np.int32)
        y_score_mat = np.zeros((len(query_labels), L), dtype=np.float32)

        # PR Curve flattened lists (收集用于AUPRC的数据)
        global_y_true_flat = []
        global_y_score_flat = []

        # 3. Iterate Queries
        for i in range(len(query_labels)):
            true_set = set(query_labels[i])
            for ec in true_set:
                j = self.label_to_idx.get(ec)
                if j is not None:
                    y_true_mat[i, j] = 1

            # --- Voting Logic ---
            counter = defaultdict(float)
            
            # Calculate Weights & Fill Counter
            if weighted:
                if weighting_mode == "softmax":
                    w_vec = torch.softmax(topk_sims[i].float(), dim=0)
                    weights_sum = 1.0 # Softmax sums to 1
                    for jpos, idx in enumerate(topk_idx[i].tolist()):
                        w = float(w_vec[jpos].item())
                        for ec in self.gallery_labels[idx]:
                            counter[ec] += w
                elif weighting_mode == "clamp":
                    weights_sum = 0.0
                    for jpos, idx in enumerate(topk_idx[i].tolist()):
                        w = float(topk_sims[i, jpos].item())
                        w = max(w, 0.0)
                        weights_sum += w
                        for ec in self.gallery_labels[idx]:
                            counter[ec] += w
                else:
                    weights_sum = 0.0
                    for jpos, idx in enumerate(topk_idx[i].tolist()):
                        w = float(topk_sims[i, jpos].item())
                        weights_sum += w
                        for ec in self.gallery_labels[idx]:
                            counter[ec] += w
                
                # Prevent division by zero for threshold calculation
                thr_ref = weights_sum if abs(weights_sum) > 1e-12 else 1e-12
            else:
                # Unweighted
                weights_sum = float(k_eff)
                for idx in topk_idx[i].tolist():
                    for ec in self.gallery_labels[idx]:
                        counter[ec] += 1.0
                thr_ref = float(k_eff)

            # -------------------------
            # Adaptive tau (Entropy-based)
            # -------------------------
            tau_i = float(vote_tau)  # 默认：不启用时就是固定值

            if bool(self.config.get("adaptive_tau", False)) and str(self.config.get("adaptive_tau_mode", "entropy")).lower() == "entropy":
                tau_min = float(self.config.get("tau_min", 0.15))
                tau_max = float(self.config.get("tau_max", 0.55))
                tau_mix = float(self.config.get("tau_mix", 0.0))
                clamp_neg = bool(self.config.get("entropy_clamp_negative", True))

                # counter -> scores
                scores = np.array(list(counter.values()), dtype=np.float32)
                if clamp_neg:
                    scores = np.maximum(scores, 0.0)

                S = float(scores.sum())
                M = int(scores.size)

                if S <= 1e-12 or M <= 1:
                    # 信息不足：采用更宽松的阈值（提升OOD recall）
                    tau_dyn = tau_min
                else:
                    p = scores / S  # 概率分布
                    H = float(-(p * np.log(p + 1e-12)).sum())            # entropy
                    H_norm = H / float(np.log(M + 1e-12))                # 归一化到[0,1]
                    H_norm = max(0.0, min(1.0, H_norm))                  # clamp

                    # 熵低(集中) -> 更自信 -> tau更高；熵高(分散) -> tau更低
                    tau_dyn = tau_min + (1.0 - H_norm) * (tau_max - tau_min)

                # 与固定 vote_tau 混合（可选）
                if tau_mix > 0.0:
                    tau_i = (1.0 - tau_mix) * tau_dyn + tau_mix * float(vote_tau)
                else:
                    tau_i = tau_dyn

                # 最终 clamp
                tau_i = max(tau_min, min(tau_max, tau_i))

            # Determine Threshold & Predictions (use tau_i)
            thr = float(tau_i) * float(thr_ref)
            pred_set = {ec for ec, score in counter.items() if score >= thr}


            if not pred_set and counter:
                pred_set = {max(counter.items(), key=lambda x: x[1])[0]}

            # --- Metrics ---
            tp = len(true_set & pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec = tp / len(true_set) if true_set else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            f1_scores.append(f1)
            precision_scores.append(prec)
            recall_scores.append(rec)

            if save_preds:
                prediction_records.append({
                    "Query_ID": query_ids[i],
                    "True_Labels": list(true_set),
                    "Predicted_Labels": list(pred_set)
                })

            # --- Score Filling (AUC & PR) ---
            # 确定归一化因子 (为了让不同样本的分数在PR曲线上可比)
            if weighted:
                # Softmax: weights_sum=1.0. Clamp/Raw: use sum.
                norm_factor = weights_sum if weights_sum > 1e-12 else 1.0
            else:
                # Unweighted: max score is k_eff
                norm_factor = float(k_eff)

            for ec, raw_val in counter.items():
                # 1. For Standard AUC (Matrix)
                j = self.label_to_idx.get(ec)
                if j is not None:
                    # 注意：这里我们存入 normalized score，这样对AUC计算更稳健
                    y_score_mat[i, j] = float(raw_val) / norm_factor
                
                # 2. For PR Curve (Flattened List) - 仅当需要时收集
                if return_raw_scores:
                    normalized_score = float(raw_val) / norm_factor
                    global_y_score_flat.append(normalized_score)
                    global_y_true_flat.append(1 if ec in true_set else 0)

        # 4. Finalize
        if save_preds:
            preds_save_path = os.path.join(
                output_dir, f"predictions_{query_set_name.replace(' ', '_').lower()}.csv"
            )
            save_detailed_predictions(prediction_records, preds_save_path)

        auc_micro, auc_macro = self._compute_multilabel_auc(y_true_mat, y_score_mat)

        results = {
            'f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
            'precision': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'recall': float(np.mean(recall_scores)) if recall_scores else 0.0,
            'auc': float(auc_micro),
            'auc_macro': float(auc_macro),
        }

        # Return raw data for plotting if requested
        if return_raw_scores:
            results['raw_data'] = (
                np.array(global_y_true_flat, dtype=np.int32), 
                np.array(global_y_score_flat, dtype=np.float32)
            )

        return results