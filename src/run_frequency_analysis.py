"""
run_frequency_analysis.py
EC频次分析主程序

Usage:
    python run_frequency_analysis.py
"""
import os
import sys
from datetime import datetime
import logging
import pandas as pd
import json

# 导入项目模块
import config as cfg
import utils

# 导入分析模块
from ec_frequency_analysis import (
    compute_ec_frequency_in_training,
    assign_frequency_bins,
    annotate_test_samples_with_bins,
    compute_binned_metrics,
    compute_bin_center_values
)

from plot_frequency_analysis import (
    plot_binned_comparison,
    plot_improvement_trend,
    plot_frequency_distribution,
    plot_all_metrics_comparison
)


def main():
    """
    主函数：执行完整的EC频次分析流程
    """
    
    # ============================================================
    # 1. 初始化
    # ============================================================
    
    # 创建输出目录
    output_dir = os.path.join(
        "experiments", 
        f"ec_frequency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(output_dir, "frequency_analysis.log")
    utils.setup_logging(log_path)
    
    logging.info("="*80)
    logging.info(" EC Frequency Analysis - Understanding CPT's Long-tail Performance ")
    logging.info("="*80)
    logging.info(f"Output directory: {output_dir}")
    
    # ============================================================
    # 2. 配置参数
    # ============================================================
    
    # ===== 修改这里：指定训练集和预测文件 =====
    
    # 训练集CSV（用于统计EC频次）
    train_csv_path = cfg.DATA_PATHS["csv_path"]  # 或者使用 train_split_70.csv
    
    # 预测结果CSV路径
    # 格式：{'模型名': {'测试集名': 'CSV路径'}}
    prediction_files = {
        'ESM_2': {
            'PRICE': 'experiments/esm/predictions_price.csv',
            'NEW': 'experiments/esm/predictions_new.csv'
        },
        'ESM_CPT': {
            'PRICE': 'experiments/cpt/predictions_price.csv',
            'NEW': 'experiments/cpt/predictions_new.csv'
        }
    }
    import pandas as pd
    df_sample = pd.read_csv(train_csv_path)
    print("Available columns:", df_sample.columns.tolist())
    print("\nFirst few rows:")
    print(df_sample.head())
    # CSV列名配置
    ec_column = 'EC number'  # 训练集中EC标签的列名
    true_label_col = 'True_Labels'  # 预测CSV中真实标签列名
    pred_label_col = 'Predicted_Labels'  # 预测CSV中预测标签列名
    separator = ';'  # 多标签分隔符
    
    # 分箱策略
    bin_strategy = 'log'  # 'log' | 'quantile' | 'custom'
    
    # 如果使用custom，定义自定义箱
    # custom_bins = [
    #     (1, 10, 'Very Rare (1-10)'),
    #     (11, 50, 'Rare (11-50)'),
    #     (51, float('inf'), 'Common (50+)')
    # ]
    custom_bins = None
    
    logging.info("\nConfiguration:")
    logging.info(f"  Training CSV: {train_csv_path}")
    logging.info(f"  Bin strategy: {bin_strategy}")
    logging.info(f"  Models to compare: {list(prediction_files.keys())}")
    
    # 验证文件存在
    if not os.path.exists(train_csv_path):
        logging.error(f"Training CSV not found: {train_csv_path}")
        sys.exit(1)
    
    for model_name, test_sets in prediction_files.items():
        for test_name, pred_path in test_sets.items():
            if not os.path.exists(pred_path):
                logging.error(f"Prediction file not found: {pred_path}")
                logging.error(f"  Model: {model_name}, Test: {test_name}")
                sys.exit(1)
    
    # ============================================================
    # 3. 步骤1：计算训练集EC频次
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 1: Computing EC Frequencies in Training Set ")
    logging.info("="*80)
    
    ec_freq_dict = compute_ec_frequency_in_training(
        train_csv_path,
        ec_column=ec_column,
        separator=separator
    )
    
    # 保存频次统计
    freq_df = pd.DataFrame([
        {'EC': ec, 'Frequency': freq}
        for ec, freq in sorted(ec_freq_dict.items(), key=lambda x: x[1], reverse=True)
    ])
    freq_save_path = os.path.join(output_dir, 'ec_frequencies.csv')
    freq_df.to_csv(freq_save_path, index=False)
    logging.info(f"\n✓ EC frequencies saved to {freq_save_path}")
    
    # ============================================================
    # 4. 步骤2：分配频次箱
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 2: Assigning Frequency Bins ")
    logging.info("="*80)
    
    ec_to_bin, bins = assign_frequency_bins(
        ec_freq_dict,
        bin_strategy=bin_strategy,
        custom_bins=custom_bins
    )
    
    # 保存EC到箱的映射
    bin_mapping_df = pd.DataFrame([
        {'EC': ec, 'Frequency': ec_freq_dict[ec], 'Bin': bin_label}
        for ec, bin_label in sorted(ec_to_bin.items(), key=lambda x: ec_freq_dict[x[0]], reverse=True)
    ])
    bin_mapping_path = os.path.join(output_dir, 'ec_bin_mapping.csv')
    bin_mapping_df.to_csv(bin_mapping_path, index=False)
    logging.info(f"\n✓ EC-to-bin mapping saved to {bin_mapping_path}")
    
    # 计算箱中心值（用于绘图）
    bin_centers = compute_bin_center_values(bins)
    
    # ============================================================
    # 5. 步骤3：标注测试集样本
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 3: Annotating Test Samples with Frequency Bins ")
    logging.info("="*80)
    
    all_annotated = {}
    
    for model_name, test_sets in prediction_files.items():
        logging.info(f"\nProcessing model: {model_name}")
        all_annotated[model_name] = {}
        
        for test_name, pred_path in test_sets.items():
            logging.info(f"  Processing test set: {test_name}")
            logging.info(f"    Loading predictions from {pred_path}")
            
            # 加载预测结果
            pred_df = pd.read_csv(pred_path)
            
            # 标注频次箱
            annotated_df = annotate_test_samples_with_bins(
                pred_df,
                ec_to_bin,
                ec_freq_dict,
                true_label_col=true_label_col,
                pred_label_col=pred_label_col,
                separator=separator
            )
            
            all_annotated[model_name][test_name] = annotated_df
            
            # 保存标注后的数据
            save_path = os.path.join(
                output_dir,
                f'annotated_{model_name}_{test_name}.csv'
            )
            annotated_df.to_csv(save_path, index=False)
            logging.info(f"    ✓ Saved annotated data to {save_path}")
    
    # ============================================================
    # 6. 步骤4：计算每个箱的性能指标
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 4: Computing Performance Metrics per Frequency Bin ")
    logging.info("="*80)
    
    all_metrics = []
    
    for model_name, test_sets in all_annotated.items():
        for test_name, annotated_df in test_sets.items():
            logging.info(f"\nComputing metrics for {model_name} - {test_name}")
            
            metrics_df = compute_binned_metrics(
                annotated_df,
                model_name=model_name,
                test_set_name=test_name,
                true_label_col=true_label_col,
                pred_label_col=pred_label_col,
                separator=separator
            )
            
            all_metrics.append(metrics_df)
            
            # 打印该模型-测试集的指标
            logging.info(f"\n{metrics_df.to_string(index=False)}")
    
    # 合并所有指标
    combined_metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # 保存指标
    metrics_save_path = os.path.join(output_dir, 'binned_metrics.csv')
    combined_metrics_df.to_csv(metrics_save_path, index=False)
    logging.info(f"\n✓ All metrics saved to {metrics_save_path}")
    
    # 打印汇总表
    logging.info("\n" + "="*80)
    logging.info(" Summary: Performance Metrics by Frequency Bin ")
    logging.info("="*80)
    print("\n" + combined_metrics_df.to_string(index=False))
    
    # ============================================================
    # 7. 步骤5：计算相对提升
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 5: Computing Relative Improvements ")
    logging.info("="*80)
    
    # 计算ESM_CPT相对于ESM_2的提升
    improvements = []
    
    models = sorted(combined_metrics_df['Model'].unique())
    if len(models) == 2:
        model_baseline, model_cpt = models
        
        for test_set in combined_metrics_df['Test_Set'].unique():
            for bin_label in combined_metrics_df['Frequency_Bin'].unique():
                baseline_row = combined_metrics_df[
                    (combined_metrics_df['Model'] == model_baseline) &
                    (combined_metrics_df['Test_Set'] == test_set) &
                    (combined_metrics_df['Frequency_Bin'] == bin_label)
                ]
                
                cpt_row = combined_metrics_df[
                    (combined_metrics_df['Model'] == model_cpt) &
                    (combined_metrics_df['Test_Set'] == test_set) &
                    (combined_metrics_df['Frequency_Bin'] == bin_label)
                ]
                
                if len(baseline_row) == 0 or len(cpt_row) == 0:
                    continue
                
                for metric in ['Precision', 'Recall', 'F1', 'Accuracy']:
                    baseline_val = baseline_row[metric].values[0]
                    cpt_val = cpt_row[metric].values[0]
                    
                    if baseline_val > 0:
                        rel_improvement = (cpt_val - baseline_val) / baseline_val * 100
                    else:
                        rel_improvement = 0
                    
                    improvements.append({
                        'Test_Set': test_set,
                        'Frequency_Bin': bin_label,
                        'Metric': metric,
                        'Baseline_Value': baseline_val,
                        'CPT_Value': cpt_val,
                        'Absolute_Improvement': cpt_val - baseline_val,
                        'Relative_Improvement_%': rel_improvement
                    })
        
        improvements_df = pd.DataFrame(improvements)
        improvements_save_path = os.path.join(output_dir, 'improvements.csv')
        improvements_df.to_csv(improvements_save_path, index=False)
        logging.info(f"\n✓ Improvements saved to {improvements_save_path}")
        
        # 打印关键发现
        logging.info("\n" + "="*80)
        logging.info(" Key Findings: F1 Improvements by Frequency Bin ")
        logging.info("="*80)
        
        f1_improvements = improvements_df[improvements_df['Metric'] == 'F1']
        print("\n" + f1_improvements.to_string(index=False))
    
    # ============================================================
    # 8. 步骤6：生成可视化
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 6: Generating Visualizations ")
    logging.info("="*80)
    
    # 确定箱的顺序
    bin_order = [b[2] for b in bins]
    
    # 图1：EC频次分布
    plot_frequency_distribution(
        ec_freq_dict,
        ec_to_bin,
        os.path.join(output_dir, 'fig1_ec_frequency_distribution.png')
    )
    
    # 图2：F1分箱对比
    plot_binned_comparison(
        combined_metrics_df,
        os.path.join(output_dir, 'fig2_f1_binned_comparison.png'),
        metric='F1',
        bin_order=bin_order
    )
    
    # 图3：Recall分箱对比
    plot_binned_comparison(
        combined_metrics_df,
        os.path.join(output_dir, 'fig3_recall_binned_comparison.png'),
        metric='Recall',
        bin_order=bin_order
    )
    
    # 图4：F1提升趋势（对数坐标）
    plot_improvement_trend(
        combined_metrics_df,
        bin_centers,
        os.path.join(output_dir, 'fig4_f1_improvement_trend.png'),
        metric='F1'
    )
    
    # 图5：所有指标对比
    plot_all_metrics_comparison(
        combined_metrics_df,
        os.path.join(output_dir, 'fig5_all_metrics_comparison.png'),
        bin_order=bin_order
    )
    
    # ============================================================
    # 9. 生成摘要报告
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 7: Generating Summary Report ")
    logging.info("="*80)
    
    summary_report = {
        'analysis_date': datetime.now().isoformat(),
        'training_set': train_csv_path,
        'total_unique_ecs': len(ec_freq_dict),
        'frequency_range': {
            'min': int(min(ec_freq_dict.values())),
            'max': int(max(ec_freq_dict.values())),
            'mean': float(pd.Series(ec_freq_dict.values()).mean()),
            'median': float(pd.Series(ec_freq_dict.values()).median())
        },
        'bin_strategy': bin_strategy,
        'bins': bins,
        'models_compared': list(prediction_files.keys()),
        'test_sets': list(set(
            test_name 
            for test_sets in prediction_files.values() 
            for test_name in test_sets.keys()
        ))
    }
    
    # 添加关键发现
    if len(models) == 2 and 'improvements_df' in locals():
        f1_impr = improvements_df[improvements_df['Metric'] == 'F1']
        
        # 找出提升最大的箱
        max_improvement_row = f1_impr.loc[f1_impr['Relative_Improvement_%'].idxmax()]
        
        summary_report['key_findings'] = {
            'max_improvement': {
                'test_set': max_improvement_row['Test_Set'],
                'bin': max_improvement_row['Frequency_Bin'],
                'improvement_%': float(max_improvement_row['Relative_Improvement_%'])
            }
        }
    
    report_path = os.path.join(output_dir, 'summary_report.json')
    with open(report_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    logging.info(f"\n✓ Summary report saved to {report_path}")
    
    # ============================================================
    # 10. 完成
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" Analysis Completed Successfully! ")
    logging.info("="*80)
    logging.info(f"\nAll outputs saved to: {output_dir}")
    logging.info("\nGenerated files:")
    logging.info("  Data:")
    logging.info("    - ec_frequencies.csv (EC频次统计)")
    logging.info("    - ec_bin_mapping.csv (EC到箱的映射)")
    logging.info("    - annotated_*.csv (标注后的测试数据)")
    logging.info("    - binned_metrics.csv (分箱性能指标)")
    logging.info("    - improvements.csv (相对提升)")
    logging.info("  Figures:")
    logging.info("    - fig1_ec_frequency_distribution.png")
    logging.info("    - fig2_f1_binned_comparison.png ⭐ 关键图")
    logging.info("    - fig3_recall_binned_comparison.png")
    logging.info("    - fig4_f1_improvement_trend.png ⭐ 趋势图")
    logging.info("    - fig5_all_metrics_comparison.png")
    logging.info("  Report:")
    logging.info("    - summary_report.json")
    logging.info("    - frequency_analysis.log")
    
    logging.info("\n" + "="*80)


if __name__ == '__main__':
    main()