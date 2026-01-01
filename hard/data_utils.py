# triplet/hard/data_utils.py (完整版)
"""
数据处理工具 - 包含突变序列加载逻辑
"""
import os
import pandas as pd
import logging
from collections import defaultdict

def load_and_create_label_map(csv_path):
    """
    从CSV文件加载数据并创建ID到EC号的映射
    
    Args:
        csv_path: CSV文件路径（TSV格式）
    
    Returns:
        id_to_ecs: dict, {seq_id: [ec1, ec2, ...]}
        all_ids: list, 所有序列ID列表
    """
    logging.info(f"Loading data from {csv_path}...")
    
    try:
        # 读取CSV文件（TSV格式）
        df = pd.read_csv(csv_path, sep='\t')
        
        # 处理EC号列（可能是分号分隔的多个EC号）
        df['EC number'] = df['EC number'].astype(str).str.split(';').apply(
            lambda x: [i.strip() for i in x if i.strip()]
        )
        
        # 创建ID到EC号的映射
        id_to_ecs = {}
        all_ids = []
        
        for _, row in df.iterrows():
            seq_id = row['Entry']
            ec_list = row['EC number']
            
            if ec_list:  # 只保留有EC号的样本
                id_to_ecs[seq_id] = ec_list
                all_ids.append(seq_id)
        
        logging.info(f"Loaded {len(all_ids)} sequences with EC annotations")
        logging.info(f"Found {len(set([ec for ecs in id_to_ecs.values() for ec in ecs]))} unique EC numbers")
        
        return id_to_ecs, all_ids
        
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise


def augment_map_with_mutants(id_to_ecs_map, embedding_dir, train_ids_orig_only):
    """
    使用突变体增强标签映射（复用原项目逻辑）
    
    Args:
        id_to_ecs_map: 原始ID到EC号的映射
        embedding_dir: embedding目录
        train_ids_orig_only: 训练集的原始ID列表
    
    Returns:
        增强后的id_to_ecs_map（包含突变体）
    """
    logging.info("Augmenting dataset with mutant sequences...")
    
    # 获取训练集的原始ID集合
    train_orig_set = set(train_ids_orig_only)
    
    # 扫描embedding目录，找出所有突变体
    mutant_count = 0
    
    if not os.path.exists(embedding_dir):
        logging.warning(f"Embedding directory not found: {embedding_dir}")
        return id_to_ecs_map
    
    for filename in os.listdir(embedding_dir):
        if not filename.endswith('.pt'):
            continue
        
        seq_id = filename.replace('.pt', '')
        
        # 检查是否是突变体（包含下划线）
        if '_' not in seq_id:
            continue
        
        # 获取原始ID
        orig_id = seq_id.split('_')[0]
        
        # 只添加训练集原始ID的突变体
        if orig_id not in train_orig_set:
            continue
        
        # 如果原始ID有EC号，将其复制给突变体
        if orig_id in id_to_ecs_map:
            id_to_ecs_map[seq_id] = id_to_ecs_map[orig_id]
            mutant_count += 1
    
    logging.info(f"Added {mutant_count} mutant sequences to the dataset")
    
    return id_to_ecs_map


def filter_ids_with_embeddings(seq_ids, embedding_dir, prefix=""):
    """
    过滤出有embedding文件的ID
    
    Args:
        seq_ids: 序列ID列表
        embedding_dir: embedding目录
        prefix: 日志前缀
    
    Returns:
        有embedding的ID列表
    """
    valid_ids = []
    missing_ids = []
    
    for seq_id in seq_ids:
        emb_path = os.path.join(embedding_dir, f"{seq_id}.pt")
        if os.path.exists(emb_path):
            valid_ids.append(seq_id)
        else:
            missing_ids.append(seq_id)
    
    if missing_ids:
        logging.warning(f"{prefix}Missing embeddings for {len(missing_ids)}/{len(seq_ids)} samples")
    
    logging.info(f"{prefix}Valid samples: {len(valid_ids)}/{len(seq_ids)}")
    
    return valid_ids