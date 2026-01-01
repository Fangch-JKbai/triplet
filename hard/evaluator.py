# triplet/hard/evaluator.py
"""
评估器 - kNN + 聚类质量
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    silhouette_score, davies_bouldin_score,
    precision_recall_curve, average_precision_score
)
from scipy.spatial.distance import pdist, cdist  # 移到文件顶部
from collections import defaultdict
import logging


class Evaluator:
    """
    多维度评估器
    
    评估指标：
    1. kNN分类准确率/F1
    2. 聚类质量（Silhouette Score, Davies-Bouldin Index）
    3. Precision-Recall曲线
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def extract_embeddings(self, dataloader):
        """提取所有样本的embedding"""
        self.model.eval()
        
        all_embeddings = []
        all_seq_ids = []
        all_primary_ecs = []
        all_ecs = []
        
        for batch in dataloader:
            embeddings_1280 = batch["embeddings"].to(self.device)
            embeddings_128 = self.model(embeddings_1280)
            
            all_embeddings.append(embeddings_128.cpu())
            all_seq_ids.extend(batch["seq_ids"])
            all_primary_ecs.extend(batch["primary_ecs"])
            all_ecs.extend(batch["all_ecs"])
        
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        return {
            "embeddings": embeddings,
            "seq_ids": all_seq_ids,
            "primary_ecs": all_primary_ecs,
            "all_ecs": all_ecs
        }
    
    def evaluate_knn(self, train_data, test_data, k_values, tau_values=None):
        """
        kNN评估
        
        Args:
            train_data: 训练集数据（dict）
            test_data: 测试集数据（dict）
            k_values: k值列表
            tau_values: 投票阈值列表（可选，用于multi-label）
        """
        train_emb = train_data["embeddings"]
        train_labels = train_data["primary_ecs"]
        test_emb = test_data["embeddings"]
        test_labels = test_data["primary_ecs"]
        
        results = {}
        best_f1 = -1
        best_config = None
        
        for k in k_values:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric='cosine',
                weights='distance',
                n_jobs=-1
            )
            knn.fit(train_emb, train_labels)
            predictions = knn.predict(test_emb)
            
            acc = accuracy_score(test_labels, predictions)
            f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
            prec = precision_score(test_labels, predictions, average='weighted', zero_division=0)
            rec = recall_score(test_labels, predictions, average='weighted', zero_division=0)
            
            results[k] = {
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec
            }
            
            if f1 > best_f1:
                best_f1 = f1
                best_config = k
        
        logging.info(f"\n=== kNN Evaluation Results ===")
        for k, metrics in results.items():
            logging.info(f"k={k}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
                        f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")
        logging.info(f"Best k={best_config}, F1={best_f1:.4f}")
        
        return results, best_config
    
    def evaluate_clustering(self, data):
        """
        评估聚类质量
        
        Args:
            data: 数据（dict）
        """
        embeddings = data["embeddings"]
        labels = data["primary_ecs"]
        
        # 需要至少2个类别
        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            return {}
        
        try:
            # Silhouette Score (越高越好，范围-1到1)
            silhouette = silhouette_score(embeddings, labels)
            
            # Davies-Bouldin Index (越低越好)
            db_index = davies_bouldin_score(embeddings, labels)
            
            # 计算类内/类间距离比
            intra_dist, inter_dist = self._compute_intra_inter_distance(embeddings, labels)
            separation_ratio = inter_dist / (intra_dist + 1e-8)
            
            results = {
                "silhouette_score": silhouette,
                "davies_bouldin_index": db_index,
                "intra_class_distance": intra_dist,
                "inter_class_distance": inter_dist,
                "separation_ratio": separation_ratio
            }
            
            logging.info(f"\n=== Clustering Quality ===")
            logging.info(f"Silhouette Score: {silhouette:.4f} (higher is better)")
            logging.info(f"Davies-Bouldin Index: {db_index:.4f} (lower is better)")
            logging.info(f"Separation Ratio: {separation_ratio:.4f} (higher is better)")
            
            return results
            
        except Exception as e:
            logging.warning(f"Failed to compute clustering metrics: {e}")
            return {}
    
    def _compute_intra_inter_distance(self, embeddings, labels):
        """计算类内和类间平均距离"""
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)
        
        # 类内距离
        intra_distances = []
        for label, indices in label_to_indices.items():
            if len(indices) < 2:
                continue
            class_emb = embeddings[indices]
            # 计算该类内所有样本对的距离
            dists = pdist(class_emb, metric='cosine')
            intra_distances.extend(dists)
        
        intra_dist = np.mean(intra_distances) if intra_distances else 0.0
        
        # 类间距离
        inter_distances = []
        unique_labels = list(label_to_indices.keys())
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                emb1 = embeddings[label_to_indices[label1]]
                emb2 = embeddings[label_to_indices[label2]]
                
                # 计算两类之间的平均距离
                dists = cdist(emb1, emb2, metric='cosine')
                inter_distances.append(dists.mean())
        
        inter_dist = np.mean(inter_distances) if inter_distances else 0.0
        
        return intra_dist, inter_dist
    
    def compute_pr_curve_data(self, train_data, test_data, k=5):
        """
        计算PR曲线数据（用于绘图）
        
        Returns:
            (y_true, y_score, auprc)
        """
        train_emb = train_data["embeddings"]
        train_labels = train_data["primary_ecs"]
        test_emb = test_data["embeddings"]
        test_labels_list = test_data["all_ecs"]  # multi-label
        
        # 构建label vocabulary
        all_labels = set()
        for labels in train_data["all_ecs"]:
            all_labels.update(labels)
        for labels in test_labels_list:
            all_labels.update(labels)
        label_to_idx = {lb: i for i, lb in enumerate(sorted(all_labels))}
        
        # kNN
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        knn.fit(train_emb, train_labels)
        
        # 获取k个最近邻的距离和索引
        distances, indices = knn.kneighbors(test_emb)
        
        # 构建y_true和y_score
        y_true_list = []
        y_score_list = []
        
        for i, true_labels in enumerate(test_labels_list):
            # 获取k个邻居的标签
            neighbor_labels = [train_labels[idx] for idx in indices[i]]
            neighbor_dists = distances[i]
            
            # 统计每个label的加权分数
            label_scores = defaultdict(float)
            for label, dist in zip(neighbor_labels, neighbor_dists):
                weight = 1.0 / (dist + 1e-8)
                label_scores[label] += weight
            
            # 归一化
            total_weight = sum(label_scores.values())
            for label in label_scores:
                label_scores[label] /= total_weight
            
            # 对每个可能的label生成一个(y_true, y_score)对
            for label in all_labels:
                y_true_list.append(1 if label in true_labels else 0)
                y_score_list.append(label_scores.get(label, 0.0))
        
        y_true = np.array(y_true_list)
        y_score = np.array(y_score_list)
        
        if len(np.unique(y_true)) > 1:
            auprc = average_precision_score(y_true, y_score)
        else:
            auprc = 0.0
        
        return y_true, y_score, auprc