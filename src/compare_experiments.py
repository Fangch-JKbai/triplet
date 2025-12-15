# compare_experiments.py
"""
对比工具：比较两个embedding的评估结果

使用方法：
    python compare_experiments.py \
        --baseline experiments/2025-01-01_baseline \
        --improved experiments/2025-01-02_finetuned \
        --output comparison_report.md
"""
import argparse
import json
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

def load_results(exp_dir: str) -> Dict:
    """加载实验的最终结果"""
    summary_path = os.path.join(exp_dir, "final_cross_validation_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Results not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)

def calculate_improvement(baseline: float, improved: float) -> Tuple[float, str]:
    """计算改进百分比"""
    if baseline == 0:
        return 0.0, "N/A"
    
    improvement = ((improved - baseline) / abs(baseline)) * 100
    sign = "↑" if improvement > 0 else "↓"
    return improvement, sign

def generate_comparison_table(baseline_results: Dict, improved_results: Dict, 
                              output_path: str):
    """生成对比表格（Markdown格式）"""
    
    with open(output_path, 'w') as f:
        f.write("# Embedding Quality Comparison Report\n\n")
        f.write("## Summary\n\n")
        
        # 对每个测试集生成对比
        test_sets = set(baseline_results.keys()) & set(improved_results.keys())
        
        for test_set in sorted(test_sets):
            f.write(f"### {test_set.upper()} Test Set\n\n")
            
            baseline_data = baseline_results[test_set]
            improved_data = improved_results[test_set]
            
            # 下游任务性能
            f.write("#### Downstream Task Performance\n\n")
            f.write("| Metric | Baseline | Improved | Δ (%) | Trend |\n")
            f.write("|--------|----------|----------|-------|-------|\n")
            
            for metric in ['f1', 'precision', 'recall', 'auc']:
                if metric in baseline_data and metric in improved_data:
                    b_mean = baseline_data[metric]['mean']
                    i_mean = improved_data[metric]['mean']
                    delta, sign = calculate_improvement(b_mean, i_mean)
                    
                    f.write(f"| {metric.upper()} | "
                           f"{b_mean:.4f} ± {baseline_data[metric]['std']:.4f} | "
                           f"{i_mean:.4f} ± {improved_data[metric]['std']:.4f} | "
                           f"{delta:+.2f}% | {sign} |\n")
            f.write("\n")
            
            # 内在质量指标
            intrinsic_metrics = [k for k in baseline_data.keys() 
                               if k.startswith('intrinsic_')]
            if intrinsic_metrics:
                f.write("#### Intrinsic Embedding Quality\n\n")
                f.write("| Metric | Baseline | Improved | Δ (%) | Trend |\n")
                f.write("|--------|----------|----------|-------|-------|\n")
                
                for metric in intrinsic_metrics:
                    if metric in baseline_data and metric in improved_data:
                        b_mean = baseline_data[metric]['mean']
                        i_mean = improved_data[metric]['mean']
                        
                        # Davies-Bouldin是越低越好
                        if 'davies' in metric.lower():
                            delta, _ = calculate_improvement(b_mean, i_mean)
                            sign = "↑" if delta < 0 else "↓"  # 反转
                        else:
                            delta, sign = calculate_improvement(b_mean, i_mean)
                        
                        metric_name = metric.replace('intrinsic_', '').replace('_', ' ').title()
                        f.write(f"| {metric_name} | "
                               f"{b_mean:.4f} ± {baseline_data[metric]['std']:.4f} | "
                               f"{i_mean:.4f} ± {improved_data[metric]['std']:.4f} | "
                               f"{delta:+.2f}% | {sign} |\n")
                f.write("\n")
            
            # 频率分层性能
            freq_metrics = ['common_accuracy', 'medium_accuracy', 'rare_accuracy']
            if any(m in baseline_data for m in freq_metrics):
                f.write("#### Performance by EC Frequency\n\n")
                f.write("| Category | Baseline | Improved | Δ (%) | Trend |\n")
                f.write("|----------|----------|----------|-------|-------|\n")
                
                for metric in freq_metrics:
                    if metric in baseline_data and metric in improved_data:
                        b_mean = baseline_data[metric]['mean']
                        i_mean = improved_data[metric]['mean']
                        delta, sign = calculate_improvement(b_mean, i_mean)
                        
                        category = metric.split('_')[0].capitalize()
                        f.write(f"| {category} | "
                               f"{b_mean:.4f} ± {baseline_data[metric]['std']:.4f} | "
                               f"{i_mean:.4f} ± {improved_data[metric]['std']:.4f} | "
                               f"{delta:+.2f}% | {sign} |\n")
                f.write("\n")
            
            f.write("---\n\n")
        
        # 总结
        f.write("## Overall Conclusion\n\n")
        f.write("### Key Improvements\n\n")
        
        # 统计所有改进项
        improvements = []
        declines = []
        
        for test_set in test_sets:
            baseline_data = baseline_results[test_set]
            improved_data = improved_results[test_set]
            
            for metric in baseline_data.keys():
                if metric in improved_data and isinstance(baseline_data[metric], dict):
                    if 'mean' in baseline_data[metric]:
                        b_mean = baseline_data[metric]['mean']
                        i_mean = improved_data[metric]['mean']
                        delta, _ = calculate_improvement(b_mean, i_mean)
                        
                        # Davies-Bouldin特殊处理
                        if 'davies' in metric.lower():
                            delta = -delta
                        
                        if delta > 0:
                            improvements.append((metric, test_set, delta))
                        elif delta < 0:
                            declines.append((metric, test_set, delta))
        
        improvements.sort(key=lambda x: x[2], reverse=True)
        declines.sort(key=lambda x: x[2])
        
        if improvements:
            f.write("**Top 5 Improvements:**\n\n")
            for i, (metric, test_set, delta) in enumerate(improvements[:5], 1):
                f.write(f"{i}. {metric.replace('_', ' ').title()} on {test_set}: "
                       f"**+{delta:.2f}%**\n")
            f.write("\n")
        
        if declines:
            f.write("**Areas of Decline:**\n\n")
            for i, (metric, test_set, delta) in enumerate(declines[:3], 1):
                f.write(f"{i}. {metric.replace('_', ' ').title()} on {test_set}: "
                       f"{delta:.2f}%\n")
            f.write("\n")
        
        # 计算平均改进
        all_deltas = [imp[2] for imp in improvements] + [dec[2] for dec in declines]
        if all_deltas:
            avg_delta = np.mean(all_deltas)
            f.write(f"**Average Improvement Across All Metrics:** {avg_delta:+.2f}%\n\n")
    
    print(f"Comparison report saved to {output_path}")

def generate_comparison_plots(baseline_results: Dict, improved_results: Dict, 
                             output_dir: str):
    """生成对比可视化图"""
    os.makedirs(output_dir, exist_ok=True)
    
    test_sets = sorted(set(baseline_results.keys()) & set(improved_results.keys()))
    
    # 图1: 下游任务性能对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Downstream Task Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['f1', 'precision', 'recall', 'auc']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        baseline_means = []
        improved_means = []
        baseline_stds = []
        improved_stds = []
        
        for test_set in test_sets:
            if metric in baseline_results[test_set] and metric in improved_results[test_set]:
                baseline_means.append(baseline_results[test_set][metric]['mean'])
                baseline_stds.append(baseline_results[test_set][metric]['std'])
                improved_means.append(improved_results[test_set][metric]['mean'])
                improved_stds.append(improved_results[test_set][metric]['std'])
        
        x = np.arange(len(test_sets))
        width = 0.35
        
        ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
               label='Baseline', alpha=0.8, capsize=5, color='steelblue')
        ax.bar(x + width/2, improved_means, width, yerr=improved_stds, 
               label='Improved', alpha=0.8, capsize=5, color='coral')
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([ts.upper() for ts in test_sets])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'downstream_comparison.png'), dpi=300)
    plt.close()
    
    # 图2: 内在质量指标对比
    intrinsic_metrics = []
    for test_set in test_sets:
        for key in baseline_results[test_set].keys():
            if key.startswith('intrinsic_') and key not in intrinsic_metrics:
                intrinsic_metrics.append(key)
    
    if intrinsic_metrics:
        n_metrics = len(intrinsic_metrics)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_metrics)
        width = 0.35
        
        # 只用第一个测试集的数据作为示例
        test_set = test_sets[0]
        baseline_means = []
        improved_means = []
        
        for metric in intrinsic_metrics:
            if metric in baseline_results[test_set] and metric in improved_results[test_set]:
                baseline_means.append(baseline_results[test_set][metric]['mean'])
                improved_means.append(improved_results[test_set][metric]['mean'])
        
        ax.bar(x - width/2, baseline_means, width, label='Baseline', 
               alpha=0.8, color='steelblue')
        ax.bar(x + width/2, improved_means, width, label='Improved', 
               alpha=0.8, color='coral')
        
        ax.set_ylabel('Score')
        ax.set_title(f'Intrinsic Quality Comparison ({test_set.upper()})')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('intrinsic_', '').replace('_', '\n') 
                           for m in intrinsic_metrics], fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'intrinsic_comparison.png'), dpi=300)
        plt.close()
    
    print(f"Comparison plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Compare two embedding experiments')
    parser.add_argument('--baseline', required=True, 
                       help='Path to baseline experiment directory')
    parser.add_argument('--improved', required=True, 
                       help='Path to improved experiment directory')
    parser.add_argument('--output', default='comparison_report.md', 
                       help='Output markdown file path')
    parser.add_argument('--plot-dir', default='comparison_plots', 
                       help='Directory to save comparison plots')
    
    args = parser.parse_args()
    
    print("Loading results...")
    baseline_results = load_results(args.baseline)
    improved_results = load_results(args.improved)
    
    print("Generating comparison table...")
    generate_comparison_table(baseline_results, improved_results, args.output)
    
    print("Generating comparison plots...")
    generate_comparison_plots(baseline_results, improved_results, args.plot_dir)
    
    print("\n✅ Comparison completed successfully!")
    print(f"   - Report: {args.output}")
    print(f"   - Plots: {args.plot_dir}/")

if __name__ == '__main__':
    main()