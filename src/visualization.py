# visualization.py
"""
可视化工具模块：生成各种图表来展示embedding质量
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter
import logging
from typing import List, Dict

# 设置中文显示（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_tsne(embeddings: np.ndarray, labels: List[str], 
              save_path: str, max_samples: int = 2000, 
              max_classes: int = 20, title: str = "t-SNE Visualization"):
    """
    绘制t-SNE降维可视化
    
    Args:
        embeddings: [N, D] 的numpy数组
        labels: 长度为N的标签列表（单标签）
        save_path: 保存路径
        max_samples: 最多采样多少个点
        max_classes: 最多显示多少个类别
    """
    logging.info(f"Generating t-SNE plot (max {max_samples} samples)...")
    
    # 采样
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]
    
    # 只保留最常见的类别
    label_counts = Counter(labels)
    top_classes = [cls for cls, _ in label_counts.most_common(max_classes)]
    
    filtered_indices = [i for i, lbl in enumerate(labels) if lbl in top_classes]
    embeddings = embeddings[filtered_indices]
    labels = [labels[i] for i in filtered_indices]
    
    if len(embeddings) < 10:
        logging.warning("Too few samples for t-SNE, skipping...")
        return
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 为每个类别分配颜色
    unique_labels = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                  c=[label_to_color[lbl]], label=lbl, alpha=0.6, s=20)
    
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"t-SNE plot saved to {save_path}")


def plot_similarity_distribution(positive_sims: List[float], 
                                 negative_sims: List[float],
                                 save_path: str, num_bins: int = 50):
    """
    绘制正样本对和负样本对的相似度分布直方图
    """
    logging.info("Generating similarity distribution plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(positive_sims, bins=num_bins, alpha=0.6, label='Positive Pairs', 
            color='green', density=True)
    ax.hist(negative_sims, bins=num_bins, alpha=0.6, label='Negative Pairs', 
            color='red', density=True)
    
    # 添加均值线
    pos_mean = np.mean(positive_sims)
    neg_mean = np.mean(negative_sims)
    ax.axvline(pos_mean, color='darkgreen', linestyle='--', 
               label=f'Positive Mean: {pos_mean:.3f}')
    ax.axvline(neg_mean, color='darkred', linestyle='--', 
               label=f'Negative Mean: {neg_mean:.3f}')
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Similarity Distribution: Positive vs Negative Pairs', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Similarity distribution plot saved to {save_path}")


def plot_tuning_heatmap(k_values: List[int], tau_values: List[float],
                       heatmap_data: np.ndarray, save_path: str,
                       best_k: int = None, best_tau: float = None):
    """
    绘制k vs tau的性能热图
    
    Args:
        k_values: k的搜索范围
        tau_values: tau的搜索范围
        heatmap_data: [len(k_values), len(tau_values)] 的性能矩阵
        best_k, best_tau: 最佳参数（如果提供，会在图上标注）
    """
    logging.info("Generating hyperparameter tuning heatmap...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[f'{tau:.2f}' for tau in tau_values],
                yticklabels=k_values, ax=ax, cbar_kws={'label': 'F1 Score'})
    
    # 标注最佳点
    if best_k is not None and best_tau is not None:
        k_idx = k_values.index(best_k)
        tau_idx = tau_values.index(best_tau)
        ax.add_patch(plt.Rectangle((tau_idx, k_idx), 1, 1, 
                                   fill=False, edgecolor='red', lw=3))
    
    ax.set_xlabel('Voting Threshold (tau)', fontsize=12)
    ax.set_ylabel('k (Number of Neighbors)', fontsize=12)
    ax.set_title('Hyperparameter Tuning: F1 Score Heatmap', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Tuning heatmap saved to {save_path}")


def plot_ec_frequency_performance(results: Dict[str, float], save_path: str):
    """
    绘制按EC频率分层的性能柱状图
    """
    logging.info("Generating EC frequency performance plot...")
    
    categories = ['common', 'medium', 'rare']
    accuracies = [results.get(f'{cat}_accuracy', 0.0) for cat in categories]
    counts = [results.get(f'{cat}_count', 0) for cat in categories]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.4
    
    # 准确率柱状图
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', 
                    color='skyblue', alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim([0, 1.0])
    
    # 样本数量柱状图（第二y轴）
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, counts, width, label='Sample Count', 
                    color='orange', alpha=0.8)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax1.set_title('Performance by EC Frequency Category', fontsize=14)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # 在柱子上标注数值
    for i, (acc, cnt) in enumerate(zip(accuracies, counts)):
        ax1.text(i - width/2, acc + 0.02, f'{acc:.3f}', 
                ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, cnt + max(counts)*0.02, f'{cnt}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"EC frequency performance plot saved to {save_path}")


def plot_knn_accuracy_curve(results: Dict[str, float], save_path: str):
    """
    绘制不同k值下的kNN准确率曲线
    """
    logging.info("Generating k-NN accuracy curve...")
    
    k_values = []
    accuracies = []
    
    for key, value in sorted(results.items()):
        if key.startswith('knn_accuracy_k'):
            k = int(key.split('knn_accuracy_k')[1])
            k_values.append(k)
            accuracies.append(value)
    
    if not k_values:
        logging.warning("No k-NN accuracy data found, skipping plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, accuracies, marker='o', linewidth=2, 
            markersize=8, color='purple', label='k-NN Accuracy')
    
    # 标注每个点的数值
    for k, acc in zip(k_values, accuracies):
        ax.text(k, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
    
    ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('k-NN Accuracy vs k', fontsize=14)
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"k-NN accuracy curve saved to {save_path}")


def create_comprehensive_report_figure(all_metrics: Dict, save_path: str):
    """
    创建一个综合报告图，包含多个子图展示不同维度的性能
    """
    logging.info("Creating comprehensive report figure...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 子图1: 下游任务性能 (F1, Precision, Recall, AUC)
    ax1 = fig.add_subplot(gs[0, 0])
    downstream_metrics = ['f1_at_fixed_tau', 'precision_at_fixed_tau', 
                         'recall_at_fixed_tau', 'auc']
    downstream_values = []
    downstream_labels = []
    for key, value in all_metrics.items():
        for metric in downstream_metrics:
            if key.startswith(metric):
                downstream_values.append(value)
                downstream_labels.append(key.replace('_at_fixed_tau', '').capitalize())
                break
    
    if downstream_values:
        ax1.bar(range(len(downstream_values)), downstream_values, color='steelblue')
        ax1.set_xticks(range(len(downstream_values)))
        ax1.set_xticklabels(downstream_labels, rotation=45, ha='right')
        ax1.set_ylim([0, 1.0])
        ax1.set_title('Downstream Task Performance', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
    
    # 子图2: 聚类质量指标
    ax2 = fig.add_subplot(gs[0, 1])
    if 'silhouette_score' in all_metrics:
        sil_score = all_metrics['silhouette_score']
        db_index = all_metrics.get('davies_bouldin_index', 0)
        
        ax2.bar(['Silhouette\nScore', 'Davies-Bouldin\nIndex (inverted)'], 
               [sil_score, 1.0 / (1.0 + db_index)],  # 反转DB使其越高越好
               color=['green', 'orange'])
        ax2.set_ylim([0, 1.0])
        ax2.set_title('Clustering Quality Metrics', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    
    # 子图3: k-NN准确率
    ax3 = fig.add_subplot(gs[1, 0])
    knn_keys = sorted([k for k in all_metrics.keys() if k.startswith('knn_accuracy_k')])
    if knn_keys:
        k_vals = [int(k.split('knn_accuracy_k')[1]) for k in knn_keys]
        knn_accs = [all_metrics[k] for k in knn_keys]
        ax3.plot(k_vals, knn_accs, marker='o', linewidth=2, color='purple')
        ax3.set_xlabel('k')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim([0, 1.0])
        ax3.set_title('k-NN Accuracy', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
    
    # 子图4: EC频率分层性能
    ax4 = fig.add_subplot(gs[1, 1])
    freq_cats = ['common', 'medium', 'rare']
    freq_accs = [all_metrics.get(f'{cat}_accuracy', 0) for cat in freq_cats]
    freq_counts = [all_metrics.get(f'{cat}_count', 0) for cat in freq_cats]
    
    x = np.arange(len(freq_cats))
    bars = ax4.bar(x, freq_accs, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.capitalize() for c in freq_cats])
    ax4.set_ylim([0, 1.0])
    ax4.set_title('Performance by EC Frequency', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注样本数
    for i, (bar, count) in enumerate(zip(bars, freq_counts)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # 子图5: 相似度分布对比
    ax5 = fig.add_subplot(gs[2, :])
    if 'positive_sim_mean' in all_metrics:
        pos_mean = all_metrics['positive_sim_mean']
        pos_std = all_metrics['positive_sim_std']
        neg_mean = all_metrics['negative_sim_mean']
        neg_std = all_metrics['negative_sim_std']
        separation = all_metrics.get('separation_score', 0)
        
        categories = ['Positive Pairs', 'Negative Pairs']
        means = [pos_mean, neg_mean]
        stds = [pos_std, neg_std]
        
        bars = ax5.bar(categories, means, yerr=stds, capsize=10, 
                      color=['green', 'red'], alpha=0.7)
        ax5.set_ylabel('Cosine Similarity')
        ax5.set_title(f'Similarity Distribution (Separation Score: {separation:.4f})', 
                     fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # 标注数值
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Comprehensive Embedding Quality Report', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comprehensive report figure saved to {save_path}")