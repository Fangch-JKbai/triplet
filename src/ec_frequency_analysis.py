"""
ec_frequency_analysis.py
EC频次分箱分析 - 核心功能模块
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import logging


def compute_ec_frequency_in_training(
    train_csv_path: str,
    ec_column: str = 'EC number',
    separator: str = '|'
) -> Dict[str, int]:
    """
    统计训练集中每个EC的出现频次
    
    Args:
        train_csv_path: 训练集CSV路径
        ec_column: EC标签列名
        separator: 多标签分隔符
    
    Returns:
        {'1.1.1.1': 150, '2.3.1.4': 5, ...}
    """
    logging.info(f"Loading training data from {train_csv_path}...")
    
    try:
        df = pd.read_csv(train_csv_path, sep='\t')
    except Exception as e:
        logging.error(f"Failed to load training CSV: {e}")
        raise
    
    if ec_column not in df.columns:
        logging.error(f"Column '{ec_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{ec_column}' not found")
    
    ec_counter = Counter()
    
    for idx, row in df.iterrows():
        ec_str = str(row[ec_column])
        
        # 处理多标签
        if separator in ec_str:
            ecs = ec_str.split(separator)
        else:
            ecs = [ec_str]
        
        for ec in ecs:
            ec = ec.strip()
            if ec and ec != 'nan':
                ec_counter[ec] += 1
    
    logging.info(f"  Total unique ECs: {len(ec_counter)}")
    logging.info(f"  Frequency range: {min(ec_counter.values())} - {max(ec_counter.values())}")
    
    return dict(ec_counter)


def assign_frequency_bins(
    ec_freq_dict: Dict[str, int],
    bin_strategy: str = 'log',
    custom_bins: List[Tuple[int, int, str]] = None
) -> Dict[str, str]:
    """
    将EC分配到频次箱
    
    Args:
        ec_freq_dict: EC频次字典
        bin_strategy: 'log' | 'quantile' | 'custom'
        custom_bins: [(min, max, label), ...] 当bin_strategy='custom'时使用
    
    Returns:
        {'1.1.1.1': 'High (50+)', '3.4.21.5': 'Rare (1-5)', ...}
    """
    
    if bin_strategy == 'log':
        # 对数分箱（类似CLEAN）
        bins = [
            (1, 5, 'Rare (1-5)'),
            (6, 20, 'Low (6-20)'),
            (21, 50, 'Medium (21-50)'),
            (51, float('inf'), 'High (50+)')
        ]
    elif bin_strategy == 'quantile':
        # 四分位数分箱
        freqs = sorted(ec_freq_dict.values())
        q1 = int(np.percentile(freqs, 25))
        q2 = int(np.percentile(freqs, 50))
        q3 = int(np.percentile(freqs, 75))
        bins = [
            (1, q1, f'Q1 (1-{q1})'),
            (q1+1, q2, f'Q2 ({q1+1}-{q2})'),
            (q2+1, q3, f'Q3 ({q2+1}-{q3})'),
            (q3+1, float('inf'), f'Q4 ({q3+1}+)')
        ]
    elif bin_strategy == 'custom':
        bins = custom_bins if custom_bins else [
            (1, 10, 'Very Rare (1-10)'),
            (11, 50, 'Rare (11-50)'),
            (51, float('inf'), 'Common (50+)')
        ]
    else:
        raise ValueError(f"Unknown bin_strategy: {bin_strategy}")
    
    ec_to_bin = {}
    for ec, freq in ec_freq_dict.items():
        assigned = False
        for min_f, max_f, label in bins:
            if min_f <= freq <= max_f:
                ec_to_bin[ec] = label
                assigned = True
                break
        
        if not assigned:
            logging.warning(f"EC {ec} with frequency {freq} not assigned to any bin")
    
    # 统计每个箱的EC数量
    bin_counts = Counter(ec_to_bin.values())
    logging.info("\nBin distribution:")
    for bin_label, count in sorted(bin_counts.items()):
        logging.info(f"  {bin_label}: {count} ECs")
    
    return ec_to_bin, bins


def annotate_test_samples_with_bins(
    predictions_df: pd.DataFrame,
    ec_to_bin: Dict[str, str],
    ec_freq_dict: Dict[str, int],
    true_label_col: str = 'True_Labels',
    pred_label_col: str = 'Predicted_Labels',
    separator: str = '|'
) -> pd.DataFrame:
    """
    给测试集样本标注其EC的频次箱
    
    处理多标签问题：取最低频次（保守策略）
    
    Args:
        predictions_df: 预测结果DataFrame
        ec_to_bin: EC到箱的映射
        ec_freq_dict: EC频次字典
        true_label_col: 真实标签列名
        pred_label_col: 预测标签列名
        separator: 标签分隔符
    
    Returns:
        添加了Frequency_Bin和Min_EC_Frequency列的DataFrame
    """
    
    df = predictions_df.copy()
    
    bins = []
    min_freqs = []
    all_ec_freqs = []
    
    for idx, row in df.iterrows():
        # 解析真实标签
        true_labels_str = str(row[true_label_col])
        
        if separator in true_labels_str:
            true_labels = [ec.strip() for ec in true_labels_str.split(separator)]
        else:
            true_labels = [true_labels_str.strip()]
        
        # 过滤空标签
        true_labels = [ec for ec in true_labels if ec and ec != 'nan']
        
        if not true_labels:
            bins.append('Unknown')
            min_freqs.append(0)
            all_ec_freqs.append([])
            continue
        
        # 获取每个EC的频次
        freqs = [ec_freq_dict.get(ec, 0) for ec in true_labels]
        all_ec_freqs.append(freqs)
        
        # 取最低频次（保守策略）
        min_freq = min(freqs) if freqs else 0
        min_freqs.append(min_freq)
        
        # 找到对应的箱（使用最低频次的EC）
        min_freq_ec = true_labels[freqs.index(min_freq)] if freqs else None
        
        if min_freq_ec and min_freq_ec in ec_to_bin:
            assigned_bin = ec_to_bin[min_freq_ec]
        else:
            assigned_bin = 'Unknown'
        
        bins.append(assigned_bin)
    
    df['Frequency_Bin'] = bins
    df['Min_EC_Frequency'] = min_freqs
    df['All_EC_Frequencies'] = all_ec_freqs
    
    logging.info(f"\nAnnotated {len(df)} samples:")
    bin_dist = Counter(bins)
    for bin_label, count in sorted(bin_dist.items()):
        logging.info(f"  {bin_label}: {count} samples")
    
    return df


def compute_binned_metrics(
    annotated_df: pd.DataFrame,
    model_name: str,
    test_set_name: str,
    true_label_col: str = 'True_Labels',
    pred_label_col: str = 'Predicted_Labels',
    separator: str = '|'
) -> pd.DataFrame:
    """
    计算每个频次箱的性能指标
    
    Returns:
        DataFrame with columns: Model, Test_Set, Frequency_Bin, N_Samples, 
                                Precision, Recall, F1, Accuracy
    """
    
    results = []
    
    for bin_label in annotated_df['Frequency_Bin'].unique():
        if bin_label == 'Unknown':
            continue
        
        subset = annotated_df[annotated_df['Frequency_Bin'] == bin_label]
        n_samples = len(subset)
        
        if n_samples == 0:
            continue
        
        # 计算指标
        precisions = []
        recalls = []
        f1s = []
        corrects = []
        
        for idx, row in subset.iterrows():
            # 解析标签
            true_str = str(row[true_label_col])
            pred_str = str(row[pred_label_col])
            
            true_set = set([ec.strip() for ec in true_str.split(separator) if ec.strip() and ec.strip() != 'nan'])
            pred_set = set([ec.strip() for ec in pred_str.split(separator) if ec.strip() and ec.strip() != 'nan'])
            
            if not true_set:
                continue
            
            # 计算TP, FP, FN
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            # Precision, Recall, F1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            
            # Accuracy（至少一个命中算对）
            is_correct = tp > 0
            corrects.append(is_correct)
        
        results.append({
            'Model': model_name,
            'Test_Set': test_set_name,
            'Frequency_Bin': bin_label,
            'N_Samples': len(corrects),
            'Precision': np.mean(precisions) if precisions else 0.0,
            'Recall': np.mean(recalls) if recalls else 0.0,
            'F1': np.mean(f1s) if f1s else 0.0,
            'Accuracy': np.mean(corrects) if corrects else 0.0
        })
    
    return pd.DataFrame(results)


def compute_bin_center_values(bins: List[Tuple[int, int, str]]) -> Dict[str, float]:
    """
    计算每个箱的中心值（用于绘图X轴）
    
    Returns:
        {'Rare (1-5)': 3.0, 'Low (6-20)': 13.0, ...}
    """
    bin_centers = {}
    
    for min_f, max_f, label in bins:
        if max_f == float('inf'):
            # 对于开区间，使用min_f的1.5倍作为中心
            center = min_f * 1.5
        else:
            # 对数中心
            center = np.sqrt(min_f * max_f)
        
        bin_centers[label] = center
    
    return bin_centers