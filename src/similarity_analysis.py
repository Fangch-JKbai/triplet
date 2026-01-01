"""
similarity_analysis.py
收集相似度分布的核心指标
"""
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.special import softmax
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging


class SimilarityAnalyzer:
    """
    收集并分析相似度分布的关键指标
    """
    
    def __init__(self, evaluator, query_loader, k=10):
        self.evaluator = evaluator
        self.query_loader = query_loader
        self.k = k
        
    def collect_similarity_metrics(self) -> pd.DataFrame:
        """
        对每个query计算相似度分布指标
        
        Returns:
            DataFrame with columns:
            - query_id
            - true_labels
            - predicted_labels
            - is_correct
            - top1_sim
            - top2_sim
            - margin
            - topk_mean
            - topk_std
            - topk_entropy
            - gap_ratio
            - topk_sims (list)
        """
        
        logging.info("Collecting query embeddings and similarities...")
        
        # 获取query embeddings
        query_emb_cpu, query_labels, query_ids = \
            self.evaluator._get_all_embeddings_and_labels(
                self.query_loader, "Collecting Query Embeddings"
            )
        
        if query_emb_cpu.numel() == 0:
            logging.warning("No query embeddings found!")
            return pd.DataFrame()
        
        # 移动到合适的设备
        query_embeddings = self.evaluator._maybe_move_query_embeddings(query_emb_cpu)
        gallery_embeddings = self.evaluator.gallery_embeddings
        
        if not self.evaluator.config["keep_gallery_on_device"]:
            gallery_embeddings = gallery_embeddings.cpu()
        
        # 计算相似度矩阵
        logging.info("Computing similarity matrix...")
        if self.evaluator.config.get("sim_dtype_fp32", True):
            sims = torch.matmul(
                query_embeddings.float(), 
                gallery_embeddings.T.float()
            )
        else:
            sims = torch.matmul(query_embeddings, gallery_embeddings.T)
        
        # 获取top-k
        k_eff = min(self.k, sims.shape[1])
        topk_sims, topk_idx = sims.topk(k_eff, largest=True, dim=1)
        
        logging.info(f"Analyzing {len(query_labels)} queries...")
        
        # 收集数据
        records = []
        
        for i in tqdm(range(len(query_labels)), desc="Processing queries"):
            true_set = set(query_labels[i])
            
            # 获取预测（使用简化的k-NN投票）
            pred_labels = self._get_predictions_from_topk(
                topk_idx[i], k=3
            )
            
            # 是否正确（只要预测中有一个真实标签就算对）
            is_correct = bool(true_set & pred_labels)
            
            # 相似度值
            sims_i = topk_sims[i].cpu().numpy()
            
            # 计算指标
            top1_sim = float(sims_i[0])
            top2_sim = float(sims_i[1]) if len(sims_i) > 1 else 0.0
            margin = top1_sim - top2_sim
            
            topk_mean = float(np.mean(sims_i))
            topk_std = float(np.std(sims_i))
            
            # 熵计算（使用softmax归一化）
            probs = softmax(sims_i)
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            
            # Gap ratio
            gap_ratio = (top1_sim - topk_mean) / (1.0 - topk_mean + 1e-12)
            
            records.append({
                'query_id': query_ids[i],
                'true_labels': '|'.join(sorted(true_set)),  # 转为字符串便于存储
                'predicted_labels': '|'.join(sorted(pred_labels)),
                'is_correct': is_correct,
                'top1_sim': top1_sim,
                'top2_sim': top2_sim,
                'margin': margin,
                'topk_mean': topk_mean,
                'topk_std': topk_std,
                'topk_entropy': entropy,
                'gap_ratio': gap_ratio,
                'topk_sims': ','.join([f'{s:.6f}' for s in sims_i])  # 转为字符串
            })
        
        df = pd.DataFrame(records)
        logging.info(f"Collected metrics for {len(df)} queries")
        logging.info(f"  Correct predictions: {df['is_correct'].sum()} ({df['is_correct'].mean()*100:.1f}%)")
        
        return df
    
    def _get_predictions_from_topk(self, topk_idx, k=3):
        """简化的预测逻辑（多数投票）"""
        counter = defaultdict(int)
        for idx in topk_idx[:k].tolist():
            for label in self.evaluator.gallery_labels[idx]:
                counter[label] += 1
        
        if not counter:
            return set()
        
        # 获取投票数最多的标签
        max_votes = max(counter.values())
        threshold = max(1, max_votes * 0.5)  # 至少要有50%的最高票数
        
        return {lb for lb, cnt in counter.items() if cnt >= threshold}


def collect_all_models_data(
    model_configs: Dict[str, Dict],
    gallery_loader: DataLoader,
    test_loaders: Dict[str, DataLoader],
    device: torch.device,
    eval_config: Dict,
    output_dir: str,
    k: int = 10
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    为所有模型和测试集收集相似度数据
    
    Args:
        model_configs: {
            'ESM_2': {'path': '...', 'model_class': EcClassifier, 'model_kwargs': {...}},
            ...
        }
    
    Returns:
        {
            'ESM_2': {'PRICE': df, 'NEW': df},
            'ESM_CPT': {'PRICE': df, 'NEW': df},
            ...
        }
    """
    
    # 导入必要的模块
    from model import EcClassifier
    from evaluation import Evaluator
    
    all_data = {}
    
    for model_name, model_config in model_configs.items():
        logging.info("="*60)
        logging.info(f"Processing {model_name}")
        logging.info("="*60)
        
        model_path = model_config['path']
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            continue
        
        # 加载模型
        logging.info(f"Loading model from {model_path}")
        model_class = model_config.get('model_class', EcClassifier)
        model_kwargs = model_config.get('model_kwargs', {})
        
        model = model_class(**model_kwargs).to(device)
        
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            logging.info("✓ Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            continue
        
        # 创建evaluator
        evaluator = Evaluator(model, gallery_loader, device, eval_config)
        
        all_data[model_name] = {}
        
        for test_name, test_loader in test_loaders.items():
            logging.info(f"\n  Analyzing {test_name} test set...")
            
            # 收集数据
            analyzer = SimilarityAnalyzer(evaluator, test_loader, k=k)
            df = analyzer.collect_similarity_metrics()
            
            if df.empty:
                logging.warning(f"    No data collected for {test_name}")
                continue
            
            # 添加元信息
            df['model'] = model_name
            df['test_set'] = test_name
            
            all_data[model_name][test_name] = df
            
            # 保存
            save_path = os.path.join(
                output_dir, 
                f"similarity_metrics_{model_name}_{test_name}.csv"
            )
            df.to_csv(save_path, index=False)
            logging.info(f"    ✓ Saved to {save_path}")
    
    return all_data