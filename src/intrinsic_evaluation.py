# intrinsic_evaluation.py
"""
Embedding内在质量评估模块
不依赖下游任务，直接评估embedding的聚类质量、可分性等
"""
import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm

class IntrinsicEmbeddingEvaluator:
    """直接评估embedding质量的工具类"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def evaluate_clustering_quality(self, embeddings: np.ndarray, 
                                    labels: List[str]) -> Dict[str, float]:
        """
        评估embedding的聚类质量
        
        Args:
            embeddings: [N, D] 的numpy数组
            labels: 长度为N的标签列表，每个元素是EC number字符串
        
        Returns:
            包含Silhouette Score和Davies-Bouldin Index的字典
        """
        if not self.config.get("clustering_metrics", False):
            return {}
        
        logging.info("Computing clustering quality metrics...")
        
        # 将多标签转换为单标签 (取第一个EC)
        single_labels = [label.split(',')[0] if ',' in label else label 
                        for label in labels]
        
        # 过滤掉只有一个样本的类别
        label_counts = Counter(single_labels)
        valid_indices = [i for i, lbl in enumerate(single_labels) 
                        if label_counts[lbl] > 1]
        
        if len(valid_indices) < 2:
            logging.warning("Not enough valid samples for clustering metrics")
            return {"silhouette_score": 0.0, "davies_bouldin_index": float('inf')}
        
        filtered_embs = embeddings[valid_indices]
        filtered_labels = [single_labels[i] for i in valid_indices]
        
        try:
            # Silhouette Score: [-1, 1]，越高越好
            sil_score = silhouette_score(filtered_embs, filtered_labels, 
                                        metric='cosine')
            
            # Davies-Bouldin Index: [0, inf)，越低越好
            db_index = davies_bouldin_score(filtered_embs, filtered_labels)
            
            logging.info(f"  Silhouette Score: {sil_score:.4f}")
            logging.info(f"  Davies-Bouldin Index: {db_index:.4f}")
            
            return {
                "silhouette_score": float(sil_score),
                "davies_bouldin_index": float(db_index)
            }
        except Exception as e:
            logging.error(f"Error computing clustering metrics: {e}")
            return {"silhouette_score": 0.0, "davies_bouldin_index": float('inf')}
    
    def evaluate_knn_accuracy(self, query_embeddings: np.ndarray,
                             query_labels: List[List[str]],
                             gallery_embeddings: np.ndarray,
                             gallery_labels: List[List[str]]) -> Dict[str, float]:
        """
        在原始embedding上直接做k-NN，评估检索准确率
        
        Args:
            query_embeddings: [N_q, D]
            query_labels: 长度N_q的列表，每个元素是EC列表
            gallery_embeddings: [N_g, D]
            gallery_labels: 长度N_g的列表
        
        Returns:
            不同k值下的准确率
        """
        knn_config = self.config.get("knn_evaluation", {})
        if not knn_config.get("enable", False):
            return {}
        
        k_values = knn_config.get("k_values", [1, 3, 5, 10])
        logging.info(f"Computing k-NN accuracy for k in {k_values}...")
        
        results = {}
        
        # 构建kNN模型 (使用余弦相似度)
        max_k = max(k_values)
        nbrs = NearestNeighbors(n_neighbors=max_k, metric='cosine', 
                               algorithm='brute', n_jobs=1)
        nbrs.fit(gallery_embeddings)
        
        # 查询
        distances, indices = nbrs.kneighbors(query_embeddings)
        
        for k in k_values:
            correct = 0
            total = 0
            
            for i, query_label_set in enumerate(query_labels):
                query_label_set = set(query_label_set)
                
                # 获取最近的k个邻居的标签
                neighbor_labels_list = [gallery_labels[idx] for idx in indices[i, :k]]
                
                # 合并所有邻居的标签
                neighbor_labels_union = set()
                for neighbor_labels in neighbor_labels_list:
                    neighbor_labels_union.update(neighbor_labels)
                
                # 判断是否命中
                if query_label_set & neighbor_labels_union:  # 有交集
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            results[f"knn_accuracy_k{k}"] = accuracy
            logging.info(f"  k={k}: {accuracy:.4f}")
        
        return results
    
    def evaluate_by_ec_frequency(self, query_embeddings: np.ndarray,
                                 query_labels: List[List[str]],
                                 gallery_embeddings: np.ndarray,
                                 gallery_labels: List[List[str]],
                                 gallery_ec_counts: Dict[str, int],
                                 k: int = 5) -> Dict[str, Dict]:
        """
        按EC频率分层评估k-NN性能
        
        Args:
            gallery_ec_counts: EC编号到其在gallery中出现次数的映射
        
        Returns:
            按频率分类的性能指标
        """
        freq_config = self.config.get("ec_frequency_analysis", {})
        if not freq_config.get("enable", False):
            return {}
        
        thresholds = freq_config.get("thresholds", {
            "common": 100, "medium": 10, "rare": 0
        })
        
        logging.info("Evaluating performance by EC frequency...")
        
        # 初始化结果容器
        results = {
            'common': {'correct': 0, 'total': 0},
            'medium': {'correct': 0, 'total': 0},
            'rare': {'correct': 0, 'total': 0}
        }
        
        # 构建kNN
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', 
                               algorithm='brute', n_jobs=1)
        nbrs.fit(gallery_embeddings)
        distances, indices = nbrs.kneighbors(query_embeddings)
        
        for i, query_label_set in enumerate(query_labels):
            query_label_set = set(query_label_set)
            
            # 确定该查询样本的频率类别（基于其主要EC）
            main_ec = list(query_label_set)[0]  # 取第一个EC作为代表
            count = gallery_ec_counts.get(main_ec, 0)
            
            if count > thresholds["common"]:
                category = 'common'
            elif count > thresholds["medium"]:
                category = 'medium'
            else:
                category = 'rare'
            
            # 检查邻居是否命中
            neighbor_labels_list = [gallery_labels[idx] for idx in indices[i]]
            neighbor_labels_union = set()
            for neighbor_labels in neighbor_labels_list:
                neighbor_labels_union.update(neighbor_labels)
            
            if query_label_set & neighbor_labels_union:
                results[category]['correct'] += 1
            results[category]['total'] += 1
        
        # 计算准确率
        summary = {}
        for category in ['common', 'medium', 'rare']:
            total = results[category]['total']
            if total > 0:
                acc = results[category]['correct'] / total
                summary[f"{category}_accuracy"] = acc
                summary[f"{category}_count"] = total
                logging.info(f"  {category.capitalize()}: {acc:.4f} ({total} samples)")
            else:
                summary[f"{category}_accuracy"] = 0.0
                summary[f"{category}_count"] = 0
        
        return summary
    
    def compute_similarity_distribution(self, query_embeddings: np.ndarray,
                                       gallery_embeddings: np.ndarray,
                                       query_labels: List[List[str]],
                                       gallery_labels: List[List[str]]) -> Dict:
        """
        计算正样本对和负样本对的相似度分布
        
        用于分析embedding是否能有效区分同类和异类
        """
        sim_config = self.config.get("similarity_distribution", {})
        if not sim_config.get("enable", False):
            return {}
        
        logging.info("Computing similarity distribution...")
        
        # 随机采样一些query (太多会很慢)
        max_samples = min(500, len(query_embeddings))
        sample_indices = np.random.choice(len(query_embeddings), 
                                         max_samples, replace=False)
        
        positive_sims = []
        negative_sims = []
        
        for i in tqdm(sample_indices, desc="Computing similarities"):
            query_emb = query_embeddings[i]
            query_label_set = set(query_labels[i])
            
            # 计算与所有gallery的相似度
            sims = np.dot(gallery_embeddings, query_emb)
            
            for j, sim in enumerate(sims):
                gallery_label_set = set(gallery_labels[j])
                
                if query_label_set & gallery_label_set:  # 有交集 -> 正样本
                    positive_sims.append(float(sim))
                else:  # 负样本
                    negative_sims.append(float(sim))
        
        # 计算分布统计量
        pos_sims_arr = np.array(positive_sims)
        neg_sims_arr = np.array(negative_sims)
        
        results = {
            "positive_sim_mean": float(np.mean(pos_sims_arr)) if len(pos_sims_arr) > 0 else 0.0,
            "positive_sim_std": float(np.std(pos_sims_arr)) if len(pos_sims_arr) > 0 else 0.0,
            "negative_sim_mean": float(np.mean(neg_sims_arr)) if len(neg_sims_arr) > 0 else 0.0,
            "negative_sim_std": float(np.std(neg_sims_arr)) if len(neg_sims_arr) > 0 else 0.0,
            "positive_sims_sample": positive_sims[:100],  # 保存一些样本用于可视化
            "negative_sims_sample": negative_sims[:100]
        }
        
        logging.info(f"  Positive pairs sim: {results['positive_sim_mean']:.4f} ± {results['positive_sim_std']:.4f}")
        logging.info(f"  Negative pairs sim: {results['negative_sim_mean']:.4f} ± {results['negative_sim_std']:.4f}")
        
        # 分离度指标 (正样本和负样本分布的差距)
        separation = results['positive_sim_mean'] - results['negative_sim_mean']
        results['separation_score'] = separation
        logging.info(f"  Separation Score: {separation:.4f}")
        
        return results
    
    def full_intrinsic_evaluation(self, query_embeddings: torch.Tensor,
                                  query_labels: List[List[str]],
                                  query_ids: List[str],
                                  gallery_embeddings: torch.Tensor,
                                  gallery_labels: List[List[str]],
                                  gallery_ids: List[str]) -> Dict:
        """
        执行完整的内在质量评估
        
        Returns:
            包含所有评估指标的字典
        """
        logging.info("="*50)
        logging.info("Starting Intrinsic Embedding Quality Evaluation")
        logging.info("="*50)
        
        # 转为numpy
        query_embs_np = query_embeddings.cpu().numpy()
        gallery_embs_np = gallery_embeddings.cpu().numpy()
        
        all_results = {}
        
        # 1. 聚类质量
        if self.config.get("clustering_metrics", False):
            # 合并query和gallery用于聚类分析
            all_embs = np.vstack([query_embs_np, gallery_embs_np])
            all_labels = [','.join(labels) for labels in query_labels + gallery_labels]
            
            clustering_results = self.evaluate_clustering_quality(all_embs, all_labels)
            all_results.update(clustering_results)
        
        # 2. k-NN准确率
        knn_results = self.evaluate_knn_accuracy(
            query_embs_np, query_labels, 
            gallery_embs_np, gallery_labels
        )
        all_results.update(knn_results)
        
        # 3. 按EC频率分层
        # 统计gallery中每个EC的频率
        gallery_ec_counts = defaultdict(int)
        for labels in gallery_labels:
            for ec in labels:
                gallery_ec_counts[ec] += 1
        
        freq_results = self.evaluate_by_ec_frequency(
            query_embs_np, query_labels,
            gallery_embs_np, gallery_labels,
            dict(gallery_ec_counts), k=5
        )
        all_results.update(freq_results)
        
        # 4. 相似度分布
        sim_dist_results = self.compute_similarity_distribution(
            query_embs_np, gallery_embs_np,
            query_labels, gallery_labels
        )
        all_results.update(sim_dist_results)
        
        logging.info("="*50)
        logging.info("Intrinsic Evaluation Completed")
        logging.info("="*50)
        
        return all_results