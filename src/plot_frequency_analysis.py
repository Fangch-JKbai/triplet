"""
plot_frequency_analysis.py
频次分析可视化
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_binned_comparison(
    metrics_df: pd.DataFrame,
    output_path: str,
    metric: str = 'F1',
    bin_order: List[str] = None
):
    """
    绘制分箱对比柱状图
    
    Args:
        metrics_df: 性能指标DataFrame
        output_path: 输出路径
        metric: 要绘制的指标 ('F1' | 'Precision' | 'Recall' | 'Accuracy')
        bin_order: 箱的顺序
    """
    logging.info(f"Generating binned comparison plot for {metric}...")
    
    if bin_order is None:
        bin_order = sorted(metrics_df['Frequency_Bin'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    test_sets = metrics_df['Test_Set'].unique()
    
    for idx, test_set in enumerate(test_sets):
        ax = axes[idx]
        
        subset = metrics_df[metrics_df['Test_Set'] == test_set]
        
        # 准备数据
        x = np.arange(len(bin_order))
        width = 0.35
        
        models = subset['Model'].unique()
        if len(models) != 2:
            logging.warning(f"Expected 2 models, found {len(models)}")
            continue
        
        model1, model2 = sorted(models)
        
        values1 = []
        values2 = []
        sample_counts = []
        
        for bin_label in bin_order:
            bin_subset = subset[subset['Frequency_Bin'] == bin_label]
            
            val1 = bin_subset[bin_subset['Model']==model1][metric].values
            val2 = bin_subset[bin_subset['Model']==model2][metric].values
            
            values1.append(val1[0] if len(val1) > 0 else 0)
            values2.append(val2[0] if len(val2) > 0 else 0)
            
            # 样本数（取平均，应该相同）
            n = bin_subset['N_Samples'].mean() if len(bin_subset) > 0 else 0
            sample_counts.append(int(n))
        
        # 绘制柱状图
        bars1 = ax.bar(x - width/2, values1, width, label=model1, 
                      alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, values2, width, label=model2, 
                      alpha=0.8, color='#e74c3c')
        
        # 标注提升百分比和样本数
        for i, (v1, v2, n) in enumerate(zip(values1, values2, sample_counts)):
            if v1 > 0:
                improvement = (v2 - v1) / v1 * 100
                y_pos = max(v1, v2) + 0.05
                
                # 提升百分比
                color = 'green' if improvement > 0 else 'red'
                ax.text(i, y_pos, f'{improvement:+.1f}%', 
                       ha='center', fontsize=10, fontweight='bold', color=color)
                
                # 样本数
                ax.text(i, -0.08, f'n={n}', ha='center', fontsize=8, color='gray')
        
        ax.set_xlabel('EC Frequency Bin', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{test_set} Test Set', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bin_order, rotation=20, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=11)
        ax.set_ylim([-0.12, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"  ✓ Saved to {output_path}")


def plot_improvement_trend(
    metrics_df: pd.DataFrame,
    bin_centers: Dict[str, float],
    output_path: str,
    metric: str = 'F1'
):
    """
    绘制提升幅度趋势图（相对于频次的对数坐标）
    """
    logging.info(f"Generating improvement trend plot for {metric}...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    test_sets = metrics_df['Test_Set'].unique()
    markers = ['o', 's', '^', 'D']
    
    for test_idx, test_set in enumerate(test_sets):
        subset = metrics_df[metrics_df['Test_Set'] == test_set]
        
        models = sorted(subset['Model'].unique())
        if len(models) != 2:
            continue
        
        model1, model2 = models
        
        x_values = []
        improvements = []
        sample_sizes = []
        
        for bin_label in sorted(subset['Frequency_Bin'].unique()):
            bin_subset = subset[subset['Frequency_Bin'] == bin_label]
            
            val1 = bin_subset[bin_subset['Model']==model1][metric].values
            val2 = bin_subset[bin_subset['Model']==model2][metric].values
            
            if len(val1) == 0 or len(val2) == 0:
                continue
            
            v1, v2 = val1[0], val2[0]
            
            if v1 > 0:
                improvement = (v2 - v1) / v1 * 100
            else:
                improvement = 0
            
            x_val = bin_centers.get(bin_label, 1)
            n = bin_subset['N_Samples'].mean()
            
            x_values.append(x_val)
            improvements.append(improvement)
            sample_sizes.append(n)
        
        if not x_values:
            continue
        
        # 绘制趋势线
        marker = markers[test_idx % len(markers)]
        ax.plot(x_values, improvements, marker=marker, linewidth=2.5, 
               markersize=10, label=test_set, alpha=0.8)
        
        # 标注样本数
        for x, y, n in zip(x_values, improvements, sample_sizes):
            ax.annotate(f'n={int(n)}', xy=(x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.6)
    
    ax.set_xscale('log')
    ax.set_xlabel('EC Frequency in Training Set (log scale)', fontsize=12)
    ax.set_ylabel(f'Relative {metric} Improvement (%)', fontsize=12)
    ax.set_title(f'{model2} vs {model1}: {metric} Improvement by EC Frequency', 
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"  ✓ Saved to {output_path}")


def plot_frequency_distribution(
    ec_freq_dict: Dict[str, int],
    ec_to_bin: Dict[str, str],
    output_path: str
):
    """
    绘制EC频次分布直方图
    """
    logging.info("Generating EC frequency distribution plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：频次直方图
    freqs = list(ec_freq_dict.values())
    
    ax1.hist(freqs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Frequency', fontsize=12)
    ax1.set_ylabel('Number of ECs', fontsize=12)
    ax1.set_title('EC Frequency Distribution', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'Total ECs: {len(freqs)}\n'
    stats_text += f'Mean: {np.mean(freqs):.1f}\n'
    stats_text += f'Median: {np.median(freqs):.1f}\n'
    stats_text += f'Max: {np.max(freqs)}'
    ax1.text(0.65, 0.95, stats_text, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # 右图：每个箱的EC数量
    bin_counts = {}
    for bin_label in set(ec_to_bin.values()):
        bin_counts[bin_label] = sum(1 for b in ec_to_bin.values() if b == bin_label)
    
    bins_sorted = sorted(bin_counts.items(), key=lambda x: list(ec_to_bin.values()).index(x[0]) if x[0] in ec_to_bin.values() else 999)
    labels = [b[0] for b in bins_sorted]
    counts = [b[1] for b in bins_sorted]
    
    ax2.bar(range(len(labels)), counts, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Frequency Bin', fontsize=12)
    ax2.set_ylabel('Number of ECs', fontsize=12)
    ax2.set_title('ECs per Frequency Bin', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for i, count in enumerate(counts):
        ax2.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"  ✓ Saved to {output_path}")


def plot_all_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: str,
    bin_order: List[str] = None
):
    """
    绘制所有指标的对比图（4个子图）
    """
    logging.info("Generating all metrics comparison plot...")
    
    if bin_order is None:
        bin_order = sorted(metrics_df['Frequency_Bin'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        test_sets = metrics_df['Test_Set'].unique()
        x = np.arange(len(bin_order))
        width = 0.15
        
        for test_idx, test_set in enumerate(test_sets):
            subset = metrics_df[metrics_df['Test_Set'] == test_set]
            models = sorted(subset['Model'].unique())
            
            for model_idx, model in enumerate(models):
                values = []
                for bin_label in bin_order:
                    bin_subset = subset[
                        (subset['Frequency_Bin'] == bin_label) &
                        (subset['Model'] == model)
                    ]
                    val = bin_subset[metric].values
                    values.append(val[0] if len(val) > 0 else 0)
                
                offset = (test_idx * len(models) + model_idx - 1.5) * width
                ax.bar(x + offset, values, width, 
                      label=f'{test_set}-{model}', alpha=0.7)
        
        ax.set_xlabel('Frequency Bin', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} by Frequency Bin', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bin_order, rotation=20, ha='right', fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"  ✓ Saved to {output_path}")