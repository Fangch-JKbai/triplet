"""
plot_similarity_analysis.py
生成相似度分析的可视化图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import logging

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_figure1_top1_similarity(data_dict, output_path):
    """
    图1：Top-1相似度分布（小提琴图）
    布局：2x2 (正确/错误 × PRICE/NEW)
    """
    logging.info("Generating Figure 1: Top-1 Similarity Distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    test_sets = ['PRICE', 'NEW']
    correctness = [True, False]
    
    for i, test_set in enumerate(test_sets):
        for j, is_correct in enumerate(correctness):
            ax = axes[j, i]
            
            # 合并所有模型的数据
            plot_data = []
            for model_name, test_data in data_dict.items():
                if test_set not in test_data:
                    continue
                df = test_data[test_set]
                subset = df[df['is_correct'] == is_correct]
                
                for sim in subset['top1_sim'].values:
                    plot_data.append({
                        'Model': model_name,
                        'Top-1 Similarity': sim
                    })
            
            if not plot_data:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                continue
            
            plot_df = pd.DataFrame(plot_data)
            
            # 绘制小提琴图
            sns.violinplot(
                data=plot_df, x='Model', y='Top-1 Similarity',
                ax=ax, inner='box', cut=0
            )
            
            # 标题
            status = "Correct" if is_correct else "Incorrect"
            ax.set_title(f'{test_set} - {status} Predictions', fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.set_ylabel('Top-1 Similarity', fontsize=11)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加均值线
            for idx, model in enumerate(plot_df['Model'].unique()):
                model_data = plot_df[plot_df['Model'] == model]['Top-1 Similarity']
                mean_val = model_data.mean()
                ax.hlines(mean_val, idx-0.4, idx+0.4, colors='red', 
                         linestyles='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved to {output_path}")


def plot_figure2_margin_comparison(data_dict, output_path):
    """
    图2：Top1-Top2 Margin对比（箱线图+显著性检验）
    """
    logging.info("Generating Figure 2: Margin Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    test_sets = ['PRICE', 'NEW']
    
    for idx, test_set in enumerate(test_sets):
        ax = axes[idx]
        
        # 准备数据
        plot_data = []
        for model_name, test_data in data_dict.items():
            if test_set not in test_data:
                continue
            df = test_data[test_set]
            for is_correct in [True, False]:
                subset = df[df['is_correct'] == is_correct]
                for margin in subset['margin'].values:
                    plot_data.append({
                        'Model': model_name,
                        'Status': 'Correct' if is_correct else 'Incorrect',
                        'Margin': margin
                    })
        
        if not plot_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            continue
        
        plot_df = pd.DataFrame(plot_data)
        
        # 绘制箱线图
        sns.boxplot(
            data=plot_df, x='Model', y='Margin', hue='Status',
            ax=ax, palette={'Correct': '#2ecc71', 'Incorrect': '#e74c3c'}
        )
        
        # 添加显著性检验
        models = plot_df['Model'].unique()
        if len(models) >= 2:
            # 对每对模型进行检验
            for status in ['Correct', 'Incorrect']:
                status_data = plot_df[plot_df['Status'] == status]
                if len(status_data) == 0:
                    continue
                
                model_list = status_data['Model'].unique()
                if len(model_list) >= 2:
                    model1, model2 = model_list[0], model_list[1]
                    
                    data1 = status_data[status_data['Model']==model1]['Margin']
                    data2 = status_data[status_data['Model']==model2]['Margin']
                    
                    if len(data1) > 0 and len(data2) > 0:
                        try:
                            _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            # 标注显著性
                            if p_value < 0.001:
                                sig = '***'
                            elif p_value < 0.01:
                                sig = '**'
                            elif p_value < 0.05:
                                sig = '*'
                            else:
                                sig = 'ns'
                            
                            y_max = plot_df['Margin'].max()
                            y_pos = y_max * (1.05 if status == 'Correct' else 1.15)
                            x_pos = 0.5 if len(models) == 2 else 1.0
                            
                            ax.text(x_pos, y_pos, f'{status}: {sig}', 
                                   ha='center', fontsize=11, fontweight='bold',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        except Exception as e:
                            logging.warning(f"Statistical test failed: {e}")
        
        ax.set_title(f'{test_set} Test Set', fontsize=14, fontweight='bold')
        ax.set_ylabel('Top1-Top2 Margin', fontsize=12)
        ax.set_xlabel('')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title='Prediction Status', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved to {output_path}")


def plot_figure3_margin_vs_accuracy(data_dict, output_path):
    """
    图3：Margin vs Accuracy散点图（关键！）
    """
    logging.info("Generating Figure 3: Margin vs Accuracy...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    test_sets = ['PRICE', 'NEW']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, test_set in enumerate(test_sets):
        ax = axes[idx]
        
        model_idx = 0
        for model_name, test_data in data_dict.items():
            if test_set not in test_data:
                continue
            
            df = test_data[test_set]
            
            # 将margin分桶，计算每个桶的准确率
            n_bins = 10
            df['margin_bin'] = pd.cut(df['margin'], bins=n_bins)
            binned = df.groupby('margin_bin', observed=True).agg({
                'is_correct': ['mean', 'count'],
                'margin': 'mean'
            }).reset_index()
            
            binned.columns = ['bin', 'accuracy', 'count', 'margin_center']
            binned = binned[binned['count'] >= 3]  # 过滤样本太少的桶
            
            if len(binned) == 0:
                continue
            
            color = colors[model_idx % len(colors)]
            
            # 散点图
            scatter = ax.scatter(
                binned['margin_center'], binned['accuracy'],
                s=binned['count']*3, alpha=0.6,
                color=color,
                label=f'{model_name} (n={len(df)})',
                edgecolors='black', linewidths=0.5
            )
            
            # 拟合曲线
            if len(binned) >= 3:
                try:
                    z = np.polyfit(binned['margin_center'], binned['accuracy'], 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(binned['margin_center'].min(), 
                                          binned['margin_center'].max(), 100)
                    ax.plot(x_smooth, p(x_smooth), '--', 
                           color=color, linewidth=2, alpha=0.8)
                except Exception as e:
                    logging.warning(f"Curve fitting failed for {model_name}-{test_set}: {e}")
            
            model_idx += 1
        
        ax.set_xlabel('Top1-Top2 Margin', fontsize=12)
        ax.set_ylabel('Prediction Accuracy', fontsize=12)
        ax.set_title(f'{test_set} Test Set', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, None])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved to {output_path}")


def plot_figure4_cdf_curves(data_dict, output_path):
    """
    图4：Top-k相似度CDF曲线
    """
    logging.info("Generating Figure 4: CDF Curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    test_sets = ['PRICE', 'NEW']
    correctness = [True, False]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, test_set in enumerate(test_sets):
        for j, is_correct in enumerate(correctness):
            ax = axes[j, i]
            
            model_idx = 0
            for model_name, test_data in data_dict.items():
                if test_set not in test_data:
                    continue
                
                df = test_data[test_set]
                subset = df[df['is_correct'] == is_correct]
                
                if len(subset) == 0:
                    continue
                
                # 展开top-k相似度
                all_sims = []
                for sims_str in subset['topk_sims']:
                    sims_list = [float(x) for x in sims_str.split(',')]
                    all_sims.extend(sims_list)
                
                if len(all_sims) == 0:
                    continue
                
                # 计算CDF
                sorted_sims = np.sort(all_sims)
                cdf = np.arange(1, len(sorted_sims)+1) / len(sorted_sims)
                
                color = colors[model_idx % len(colors)]
                ax.plot(sorted_sims, cdf, linewidth=2.5, 
                       label=f'{model_name} (n={len(subset)})',
                       color=color, alpha=0.8)
                
                model_idx += 1
            
            status = "Correct" if is_correct else "Incorrect"
            ax.set_title(f'{test_set} - {status}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Similarity', fontsize=11)
            ax.set_ylabel('Cumulative Probability', fontsize=11)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved to {output_path}")


def plot_figure5_entropy_comparison(data_dict, output_path):
    """
    图5：熵分布对比（小提琴图）
    """
    logging.info("Generating Figure 5: Entropy Distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    test_sets = ['PRICE', 'NEW']
    correctness = [True, False]
    
    for i, test_set in enumerate(test_sets):
        for j, is_correct in enumerate(correctness):
            ax = axes[j, i]
            
            # 合并所有模型的数据
            plot_data = []
            for model_name, test_data in data_dict.items():
                if test_set not in test_data:
                    continue
                df = test_data[test_set]
                subset = df[df['is_correct'] == is_correct]
                
                for entropy in subset['topk_entropy'].values:
                    plot_data.append({
                        'Model': model_name,
                        'Entropy': entropy
                    })
            
            if not plot_data:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                continue
            
            plot_df = pd.DataFrame(plot_data)
            
            # 绘制小提琴图
            sns.violinplot(
                data=plot_df, x='Model', y='Entropy',
                ax=ax, inner='box', cut=0
            )
            
            # 标题
            status = "Correct" if is_correct else "Incorrect"
            ax.set_title(f'{test_set} - {status} Predictions', fontsize=13, fontweight='bold')
            ax.set_ylabel('Top-k Similarity Entropy', fontsize=11)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved to {output_path}")


def plot_figure6_gap_ratio_vs_f1(data_dict, output_path):
    """
    图6：Gap Ratio vs F1-Score
    """
    logging.info("Generating Figure 6: Gap Ratio vs F1...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    plot_data = []
    model_idx = 0
    
    for model_name, test_data in data_dict.items():
        for test_set, df in test_data.items():
            # 计算F1
            accuracy = df['is_correct'].mean()
            
            # 计算平均gap ratio
            gap_ratio_mean = df['gap_ratio'].mean()
            
            # 样本数
            n_samples = len(df)
            
            color = colors[model_idx % len(colors)]
            marker = markers[model_idx % len(markers)]
            
            ax.scatter(
                gap_ratio_mean, accuracy,
                s=n_samples*0.5, alpha=0.7,
                color=color, marker=marker,
                edgecolors='black', linewidths=1.5,
                label=f'{model_name}-{test_set}'
            )
            
            plot_data.append({
                'Model': model_name,
                'Test Set': test_set,
                'Gap Ratio': gap_ratio_mean,
                'Accuracy': accuracy,
                'N': n_samples
            })
        
        model_idx += 1
    
    # 拟合整体趋势
    if len(plot_data) >= 3:
        df_plot = pd.DataFrame(plot_data)
        try:
            z = np.polyfit(df_plot['Gap Ratio'], df_plot['Accuracy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_plot['Gap Ratio'].min(), 
                                df_plot['Gap Ratio'].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.5,
                   label=f'Trend (R²={np.corrcoef(df_plot["Gap Ratio"], df_plot["Accuracy"])[0,1]**2:.3f})')
        except:
            pass
    
    ax.set_xlabel('Average Gap Ratio', fontsize=12)
    ax.set_ylabel('Prediction Accuracy', fontsize=12)
    ax.set_title('Gap Ratio vs Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved to {output_path}")


def generate_summary_table(data_dict) -> pd.DataFrame:
    """
    生成统计摘要表
    """
    logging.info("Generating summary statistics table...")
    
    summary = []
    
    for model_name, test_data in data_dict.items():
        for test_set, df in test_data.items():
            for is_correct in [True, False]:
                subset = df[df['is_correct'] == is_correct]
                
                if len(subset) == 0:
                    continue
                
                summary.append({
                    'Model': model_name,
                    'Test_Set': test_set,
                    'Status': 'Correct' if is_correct else 'Incorrect',
                    'N': len(subset),
                    'Top1_Mean': subset['top1_sim'].mean(),
                    'Top1_Std': subset['top1_sim'].std(),
                    'Margin_Mean': subset['margin'].mean(),
                    'Margin_Std': subset['margin'].std(),
                    'TopK_Mean': subset['topk_mean'].mean(),
                    'TopK_Std': subset['topk_std'].mean(),
                    'Entropy_Mean': subset['topk_entropy'].mean(),
                    'Entropy_Std': subset['topk_entropy'].std(),
                    'GapRatio_Mean': subset['gap_ratio'].mean(),
                    'GapRatio_Std': subset['gap_ratio'].std()
                })
    
    summary_df = pd.DataFrame(summary)
    
    # 格式化数值为3位小数
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'N':
            summary_df[col] = summary_df[col].apply(lambda x: f'{x:.4f}')
    
    logging.info(f"  ✓ Generated summary for {len(summary_df)} groups")
    
    return summary_df