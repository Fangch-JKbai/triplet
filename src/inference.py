# inference.py - 纯推理脚本（支持诊断分析）
"""
纯推理脚本 - 跳过训练，直接评估
功能：
1. 加载预训练模型
2. 在多个测试集上评估
3. 生成详细诊断报告（tau分布、分数区间准确率等）
4. 绘制可视化分析图
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

# 导入项目模块
import config as cfg
import data_utils
import utils
from model import EcClassifier
from datasets import EvaluationDataset
from evaluation import evaluation_collate_fn, Evaluator


class DiagnosticEvaluator(Evaluator):
    """
    扩展版Evaluator，增加诊断功能
    """
    
    @torch.no_grad()
    def evaluate_with_diagnostics(
        self,
        query_loader: DataLoader,
        k: int,
        vote_tau: float,
        query_set_name: str = "Query Set",
        save_preds: bool = True,
        output_dir: str = "."
    ) -> Dict:
        """
        评估并返回详细诊断信息
        """
        
        if self.gallery_embeddings.numel() == 0:
            logging.warning("Cannot evaluate because the gallery is empty.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        query_emb_cpu, query_labels, query_ids = self._get_all_embeddings_and_labels(
            query_loader, f"Processing {query_set_name}"
        )
        
        if query_emb_cpu.numel() == 0:
            logging.warning(f"Query set '{query_set_name}' is empty.")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        self._ensure_label_vocab_from_labels(query_labels)
        L = len(self.idx_to_label)
        
        # 计算相似度
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
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        topk_sims, topk_idx = sims.topk(k_eff, largest=True, dim=1)
        
        # 配置
        weighted = bool(self.config["weighted"])
        weighting_mode = str(self.config.get("weighting_mode", "softmax")).lower()
        
        # 指标列表
        f1_scores, precision_scores, recall_scores = [], [], []
        prediction_records = []
        
        # === 诊断数据收集 ===
        tau_distribution = []
        entropy_distribution = []
        pred_count_distribution = []
        score_bucket_stats = {
            '0.00-0.15': {'correct': 0, 'total': 0},
            '0.15-0.20': {'correct': 0, 'total': 0},
            '0.20-0.30': {'correct': 0, 'total': 0},
            '0.30+': {'correct': 0, 'total': 0}
        }
        per_sample_diagnostics = []
        
        # 迭代查询
        for i in range(len(query_labels)):
            true_set = set(query_labels[i])
            
            # 投票逻辑
            counter = defaultdict(float)
            
            if weighted:
                if weighting_mode == "softmax":
                    w_vec = torch.softmax(topk_sims[i].float(), dim=0)
                    weights_sum = 1.0
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
                else:  # raw
                    weights_sum = 0.0
                    for jpos, idx in enumerate(topk_idx[i].tolist()):
                        w = float(topk_sims[i, jpos].item())
                        weights_sum += w
                        for ec in self.gallery_labels[idx]:
                            counter[ec] += w
                
                thr_ref = weights_sum if abs(weights_sum) > 1e-12 else 1e-12
            else:
                weights_sum = float(k_eff)
                for idx in topk_idx[i].tolist():
                    for ec in self.gallery_labels[idx]:
                        counter[ec] += 1.0
                thr_ref = float(k_eff)
            
            # 动态tau计算
            tau_i = float(vote_tau)
            H_norm = 0.0  # 默认熵
            
            if bool(self.config.get("adaptive_tau", False)) and \
               str(self.config.get("adaptive_tau_mode", "entropy")).lower() == "entropy":
                tau_min = float(self.config.get("tau_min", 0.15))
                tau_max = float(self.config.get("tau_max", 0.55))
                tau_mix = float(self.config.get("tau_mix", 0.0))
                clamp_neg = bool(self.config.get("entropy_clamp_negative", True))
                
                scores = np.array(list(counter.values()), dtype=np.float32)
                if clamp_neg:
                    scores = np.maximum(scores, 0.0)
                
                S = float(scores.sum())
                M = int(scores.size)
                
                if S <= 1e-12 or M <= 1:
                    tau_dyn = tau_min
                    H_norm = 1.0  # 信息不足，标记为高熵
                else:
                    p = scores / S
                    H = float(-(p * np.log(p + 1e-12)).sum())
                    H_norm = H / float(np.log(M + 1e-12))
                    H_norm = max(0.0, min(1.0, H_norm))
                    
                    tau_dyn = tau_min + (1.0 - H_norm) * (tau_max - tau_min)
                
                if tau_mix > 0.0:
                    tau_i = (1.0 - tau_mix) * tau_dyn + tau_mix * float(vote_tau)
                else:
                    tau_i = tau_dyn
                
                tau_i = max(tau_min, min(tau_max, tau_i))
            
            # 确定预测
            thr = float(tau_i) * float(thr_ref)
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
            
            # === 诊断数据记录 ===
            tau_distribution.append(tau_i)
            entropy_distribution.append(H_norm)
            pred_count_distribution.append(len(pred_set))
            
            # 分析每个预测标签的分数区间
            norm_factor = weights_sum if weights_sum > 1e-12 else 1.0
            for ec, raw_score in counter.items():
                normalized_score = float(raw_score) / norm_factor
                is_correct = (ec in true_set)
                
                # 分类到分数区间
                if normalized_score < 0.15:
                    bucket = '0.00-0.15'
                elif normalized_score < 0.20:
                    bucket = '0.15-0.20'
                elif normalized_score < 0.30:
                    bucket = '0.20-0.30'
                else:
                    bucket = '0.30+'
                
                score_bucket_stats[bucket]['total'] += 1
                if is_correct:
                    score_bucket_stats[bucket]['correct'] += 1
            
            # 记录单样本诊断
            per_sample_diagnostics.append({
                'query_id': query_ids[i],
                'tau': tau_i,
                'entropy': H_norm,
                'pred_count': len(pred_set),
                'true_count': len(true_set),
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'top_scores': sorted(counter.items(), key=lambda x: x[1], reverse=True)[:5]
            })
            
            if save_preds:
                prediction_records.append({
                    "Query_ID": query_ids[i],
                    "True_Labels": list(true_set),
                    "Predicted_Labels": list(pred_set),
                    "Tau": tau_i,
                    "Entropy": H_norm
                })
        
        # 保存预测
        if save_preds:
            preds_save_path = os.path.join(
                output_dir, f"predictions_{query_set_name.replace(' ', '_').lower()}.csv"
            )
            utils.save_detailed_predictions(prediction_records, preds_save_path)
        
        # 计算整体指标
        results = {
            'f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
            'precision': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'recall': float(np.mean(recall_scores)) if recall_scores else 0.0,
        }
        
        # === 诊断统计 ===
        diagnostics = {
            'tau_mean': float(np.mean(tau_distribution)),
            'tau_std': float(np.std(tau_distribution)),
            'tau_min': float(np.min(tau_distribution)),
            'tau_max': float(np.max(tau_distribution)),
            'entropy_mean': float(np.mean(entropy_distribution)),
            'entropy_std': float(np.std(entropy_distribution)),
            'pred_count_mean': float(np.mean(pred_count_distribution)),
            'pred_count_std': float(np.std(pred_count_distribution)),
            'score_bucket_accuracy': {},
            'tau_distribution': tau_distribution,
            'entropy_distribution': entropy_distribution,
            'per_sample': per_sample_diagnostics
        }
        
        # 计算各分数区间准确率
        for bucket, stats in score_bucket_stats.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                diagnostics['score_bucket_accuracy'][bucket] = {
                    'accuracy': float(acc),
                    'correct': stats['correct'],
                    'total': stats['total']
                }
            else:
                diagnostics['score_bucket_accuracy'][bucket] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0
                }
        
        results['diagnostics'] = diagnostics
        
        return results


def plot_diagnostic_figures(all_results: Dict, output_dir: str):
    """
    绘制诊断图表
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Tau分布直方图（多测试集对比）
    ax1 = plt.subplot(2, 3, 1)
    for name, res in all_results.items():
        if 'diagnostics' in res:
            tau_dist = res['diagnostics']['tau_distribution']
            ax1.hist(tau_dist, bins=30, alpha=0.5, label=name)
    ax1.set_xlabel('Tau Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Tau Distribution Across Test Sets', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 熵分布直方图
    ax2 = plt.subplot(2, 3, 2)
    for name, res in all_results.items():
        if 'diagnostics' in res:
            entropy_dist = res['diagnostics']['entropy_distribution']
            ax2.hist(entropy_dist, bins=30, alpha=0.5, label=name)
    ax2.set_xlabel('Normalized Entropy', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Entropy Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Tau vs Precision散点图
    ax3 = plt.subplot(2, 3, 3)
    for name, res in all_results.items():
        if 'diagnostics' in res:
            per_sample = res['diagnostics']['per_sample']
            taus = [s['tau'] for s in per_sample]
            precs = [s['precision'] for s in per_sample]
            ax3.scatter(taus, precs, alpha=0.3, s=10, label=name)
    ax3.set_xlabel('Tau', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Tau vs Precision', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Tau vs Recall散点图
    ax4 = plt.subplot(2, 3, 4)
    for name, res in all_results.items():
        if 'diagnostics' in res:
            per_sample = res['diagnostics']['per_sample']
            taus = [s['tau'] for s in per_sample]
            recs = [s['recall'] for s in per_sample]
            ax4.scatter(taus, recs, alpha=0.3, s=10, label=name)
    ax4.set_xlabel('Tau', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.set_title('Tau vs Recall', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 分数区间准确率柱状图
    ax5 = plt.subplot(2, 3, 5)
    buckets = ['0.00-0.15', '0.15-0.20', '0.20-0.30', '0.30+']
    x = np.arange(len(buckets))
    width = 0.25
    
    for idx, (name, res) in enumerate(all_results.items()):
        if 'diagnostics' in res:
            accs = [res['diagnostics']['score_bucket_accuracy'][b]['accuracy'] 
                   for b in buckets]
            ax5.bar(x + idx*width, accs, width, label=name, alpha=0.8)
    
    ax5.set_xlabel('Score Bucket', fontsize=12)
    ax5.set_ylabel('Accuracy', fontsize=12)
    ax5.set_title('Accuracy by Score Bucket', fontsize=14)
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(buckets, rotation=15)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Entropy vs F1散点图
    ax6 = plt.subplot(2, 3, 6)
    for name, res in all_results.items():
        if 'diagnostics' in res:
            per_sample = res['diagnostics']['per_sample']
            entropies = [s['entropy'] for s in per_sample]
            f1s = [s['f1'] for s in per_sample]
            ax6.scatter(entropies, f1s, alpha=0.3, s=10, label=name)
    ax6.set_xlabel('Normalized Entropy', fontsize=12)
    ax6.set_ylabel('F1 Score', fontsize=12)
    ax6.set_title('Entropy vs F1', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "diagnostic_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Diagnostic figures saved to: {save_path}")


def print_diagnostic_summary(results: Dict, test_set_name: str):
    """
    打印诊断摘要
    """
    if 'diagnostics' not in results:
        return
    
    diag = results['diagnostics']
    
    logging.info(f"\n{'='*60}")
    logging.info(f" Diagnostic Summary for {test_set_name}")
    logging.info(f"{'='*60}")
    
    # Tau统计
    logging.info(f"\nTau Statistics:")
    logging.info(f"  Mean: {diag['tau_mean']:.4f}")
    logging.info(f"  Std:  {diag['tau_std']:.4f}")
    logging.info(f"  Min:  {diag['tau_min']:.4f}")
    logging.info(f"  Max:  {diag['tau_max']:.4f}")
    
    # 熵统计
    logging.info(f"\nEntropy Statistics:")
    logging.info(f"  Mean: {diag['entropy_mean']:.4f}")
    logging.info(f"  Std:  {diag['entropy_std']:.4f}")
    
    # 预测数量统计
    logging.info(f"\nPrediction Count Statistics:")
    logging.info(f"  Mean: {diag['pred_count_mean']:.2f}")
    logging.info(f"  Std:  {diag['pred_count_std']:.2f}")
    
    # 分数区间准确率
    logging.info(f"\nAccuracy by Score Bucket:")
    for bucket in ['0.00-0.15', '0.15-0.20', '0.20-0.30', '0.30+']:
        stats = diag['score_bucket_accuracy'][bucket]
        logging.info(f"  {bucket}: {stats['accuracy']:.2%} "
                    f"({stats['correct']}/{stats['total']})")
    
    logging.info(f"{'='*60}\n")


def main():
    # ==========================================================
    # 配置参数
    # ==========================================================
    
    # ===== 修改这里：指定你的模型路径 =====
    MODEL_PATH = "experiments/cpt/best_model.pth"
    
    # ===== 推理输出目录 =====
    inference_output_dir = os.path.join(
        cfg.ARTIFACTS_CONFIG['output_dir'], 
        f"inference_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(inference_output_dir, exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(inference_output_dir, "inference.log")
    utils.setup_logging(log_path)
    
    logging.info("="*60)
    logging.info(" Pure Inference Mode - Loading Pre-trained Model ")
    logging.info("="*60)
    logging.info(f"Model path: {MODEL_PATH}")
    logging.info(f"Output directory: {inference_output_dir}")
    logging.info(f"Device: {cfg.DEVICE}")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        logging.error("Please specify the correct model path in MODEL_PATH variable.")
        return
    
    # ==========================================================
    # 加载数据
    # ==========================================================
    logging.info("\nLoading data...")
    
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])
    
    # 加载外部测试集标签
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master.update(id_to_ecs_ext)
    
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    # 增强训练数据（用于Gallery）
    id_to_ecs_master = data_utils.augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=cfg.DATA_PATHS["embedding_dir"],
        train_ids_orig_only=all_original_ids
    )
    
    mutants_train = [sid for sid in id_to_ecs_master.keys() 
                    if '_' in sid and sid.split('_')[0] in set(all_original_ids)]
    
    train_pool_ids_raw = all_original_ids + mutants_train
    
    # 构建Gallery（增强版）
    logging.info("Building augmented gallery for retrieval...")
    gallery_ids = data_utils.filter_ids_with_embeddings(
        train_pool_ids_raw, 
        cfg.DATA_PATHS["embedding_dir"], 
        "[Gallery-Augmented] "
    )
    
    # ==========================================================
    # 创建DataLoader
    # ==========================================================
    logging.info("Creating DataLoaders...")
    
    batch_size = 64  # 推理时可以用更大的batch
    
    gallery_dataset = EvaluationDataset(
        gallery_ids, 
        id_to_ecs_master, 
        cfg.DATA_PATHS["embedding_dir"]
    )
    gallery_loader = DataLoader(
        gallery_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=evaluation_collate_fn, 
        **cfg.DATALOADER_CONFIG
    )
    
    # 创建测试集DataLoader
    ood_test_loaders = {}
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        _, ext_test_ids = data_utils.load_and_create_label_map(path)
        ext_test_dataset = EvaluationDataset(
            ext_test_ids, 
            id_to_ecs_master, 
            cfg.DATA_PATHS["embedding_dir"], 
            strict_embeddings=False
        )
        ood_test_loaders[name] = DataLoader(
            ext_test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=evaluation_collate_fn, 
            **cfg.DATALOADER_CONFIG
        )
    
    # ==========================================================
    # 加载模型
    # ==========================================================
    logging.info("\nInitializing and loading model...")
    
    model = EcClassifier(**cfg.MODEL_CONFIG).to(cfg.DEVICE)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(state_dict)
        logging.info("✓ Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    model.eval()
    
    # ==========================================================
    # 创建Evaluator（带诊断功能）
    # ==========================================================
    logging.info("Creating diagnostic evaluator...")
    
    evaluator = DiagnosticEvaluator(
        model, 
        gallery_loader, 
        cfg.DEVICE, 
        cfg.EVAL_CONFIG
    )
    
    # 获取评估超参数
    k = cfg.EVAL_CONFIG.get('default_k', 3)
    tau = cfg.EVAL_CONFIG.get('default_tau', 0.20)
    
    logging.info(f"\nEvaluation Configuration:")
    logging.info(f"  k: {k}")
    logging.info(f"  base_tau: {tau}")
    logging.info(f"  adaptive_tau: {cfg.EVAL_CONFIG.get('adaptive_tau', False)}")
    if cfg.EVAL_CONFIG.get('adaptive_tau', False):
        logging.info(f"  tau_min: {cfg.EVAL_CONFIG.get('tau_min', 0.15)}")
        logging.info(f"  tau_max: {cfg.EVAL_CONFIG.get('tau_max', 0.35)}")
        logging.info(f"  tau_mix: {cfg.EVAL_CONFIG.get('tau_mix', 0.0)}")
    
    # ==========================================================
    # 在所有测试集上评估
    # ==========================================================
    logging.info("\n" + "="*60)
    logging.info(" Starting Evaluation on Test Sets ")
    logging.info("="*60)
    
    all_results = {}
    
    for name, loader in ood_test_loaders.items():
        logging.info(f"\n{'─'*60}")
        logging.info(f" Evaluating on: {name.upper()} ")
        logging.info(f"{'─'*60}")
        
        results = evaluator.evaluate_with_diagnostics(
            loader, 
            k=k, 
            vote_tau=tau,
            query_set_name=name.upper(), 
            save_preds=True, 
            output_dir=inference_output_dir
        )
        
        all_results[name.upper()] = results
        
        # 打印基础指标
        logging.info(f"\n{name.upper()} Results:")
        logging.info(f"  F1:        {results['f1']:.4f}")
        logging.info(f"  Precision: {results['precision']:.4f}")
        logging.info(f"  Recall:    {results['recall']:.4f}")
        
        # 打印诊断摘要
        print_diagnostic_summary(results, name.upper())
    
    # ==========================================================
    # 保存结果
    # ==========================================================
    results_to_save = {}
    for name, res in all_results.items():
        # 移除诊断数据中的大数组（只保留统计量）
        results_to_save[name] = {
            'f1': res['f1'],
            'precision': res['precision'],
            'recall': res['recall'],
            'diagnostics_summary': {
                'tau_mean': res['diagnostics']['tau_mean'],
                'tau_std': res['diagnostics']['tau_std'],
                'entropy_mean': res['diagnostics']['entropy_mean'],
                'pred_count_mean': res['diagnostics']['pred_count_mean'],
                'score_bucket_accuracy': res['diagnostics']['score_bucket_accuracy']
            }
        }
    
    results_path = os.path.join(inference_output_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    logging.info(f"\nResults saved to: {results_path}")
    
    # ==========================================================
    # 绘制诊断图表
    # ==========================================================
    logging.info("\nGenerating diagnostic visualizations...")
    plot_diagnostic_figures(all_results, inference_output_dir)
    
    # ==========================================================
    # 保存详细诊断数据（可选）
    # ==========================================================
    detailed_diagnostics_path = os.path.join(
        inference_output_dir, 
        "detailed_diagnostics.json"
    )
    
    # 只保存per_sample诊断（不含大数组）
    detailed_diag = {}
    for name, res in all_results.items():
        if 'diagnostics' in res:
            detailed_diag[name] = {
                'per_sample': res['diagnostics']['per_sample'][:100]  # 只保存前100个样本
            }
    
    with open(detailed_diagnostics_path, 'w') as f:
        json.dump(detailed_diag, f, indent=2)
    logging.info(f"Detailed diagnostics (first 100 samples) saved to: {detailed_diagnostics_path}")
    
    # ==========================================================
    # 完成
    # ==========================================================
    logging.info("\n" + "="*60)
    logging.info(" Inference Completed Successfully ")
    logging.info("="*60)
    logging.info(f"\nAll artifacts saved to: {inference_output_dir}")
    logging.info(f"  - Predictions: predictions_*.csv")
    logging.info(f"  - Results: inference_results.json")
    logging.info(f"  - Diagnostics: diagnostic_analysis.png")
    logging.info(f"  - Detailed diagnostics: detailed_diagnostics.json")
    logging.info(f"  - Log: inference.log")


if __name__ == '__main__':
    main()