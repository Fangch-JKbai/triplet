# coding: utf-8
"""
决策边界与置信度分布可视化实验
目标：证明对比学习改善了决策边界的清晰度

(版本 3 - [最终版] 保留小提琴图，但更新标签为“干扰项中位数”)

使用方法：
1. 仔细核对 'MODEL_CONFIGS' 字典中的所有路径
2. 运行: python decision_boundary_experiment.py
3. 查看结果: decision_boundary_experiment/decision_boundary_comparison.png
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import logging

# 导入您的项目模块
import config as cfg
import data_utils
from model import EcClassifier
from datasets import EvaluationDataset
from evaluation import evaluation_collate_fn


class PrototypeBasedEvaluator:
    """
    基于原型（Prototype）的评估器
    核心思想：每个EC类别用其所有Gallery样本的平均嵌入表示
    相似度分数即为"伪Logits"
    """
    def __init__(self, model, gallery_loader, device):
        self.model = model
        self.device = device
        
        logging.info("Building EC prototypes from gallery...")
        self.prototypes, self.ec_list = self._build_prototypes(gallery_loader)
        self.num_classes = len(self.ec_list)
        logging.info(f"Gallery built successfully with {self.num_classes} EC prototypes.")
        
    @torch.no_grad()
    def _build_prototypes(self, gallery_loader):
        """
        构建每个EC类别的原型向量（均值嵌入）
        
        返回：
            prototypes: [num_classes, 128] 的tensor
            ec_list: EC号的列表（按索引对应）
        """
        self.model.eval()
        
        # 收集所有gallery样本的嵌入和标签
        ec_embeddings = defaultdict(list)  # ec -> [嵌入列表]
        
        for batch in tqdm(gallery_loader, desc="Extracting Gallery Embeddings"):
            if batch.get("empty", False):
                continue
                
            embeddings_1280 = batch['embedding_1280'].to(self.device)
            embeddings_128 = self.model(embeddings_1280)  # [B, 128] 归一化嵌入
            
            labels_list = batch['labels']  # List[List[str]]
            
            # 将每个样本的嵌入分配给其所有EC类别
            for i, ec_labels in enumerate(labels_list):
                emb = embeddings_128[i].cpu()
                for ec in ec_labels:
                    ec_embeddings[ec].append(emb)
        
        # 计算每个EC的原型（均值后再归一化）
        ec_list = sorted(ec_embeddings.keys())
        prototypes_list = []
        
        for ec in ec_list:
            if not ec_embeddings[ec]: # 以防万一
                continue
            embs = torch.stack(ec_embeddings[ec])  # [N, 128]
            prototype = embs.mean(dim=0)  # [128]
            # 重新归一化原型向量
            prototype = torch.nn.functional.normalize(prototype, p=2, dim=0)
            prototypes_list.append(prototype)
        
        prototypes = torch.stack(prototypes_list)  # [num_classes, 128]
        return prototypes, ec_list
    
    @torch.no_grad()
    def extract_logits_for_queries(self, query_loader):
        """
        为查询集中的每个样本提取"伪Logits"（原型相似度）
        
        返回：
            results: List[Dict] 包含：
                - seq_id: 序列ID
                - true_labels: List[str] 真实标签
                - logits: [num_classes] 的numpy数组（余弦相似度分数）
        """
        self.model.eval()
        results = []
        
        for batch in tqdm(query_loader, desc="Computing Logits for Queries"):
            if batch.get("empty", False):
                continue
            
            embeddings_1280 = batch['embedding_1280'].to(self.device)
            embeddings_128 = self.model(embeddings_1280)  # [B, 128]
            
            # 计算与所有原型的余弦相似度（已归一化，直接点积）
            logits = torch.matmul(embeddings_128, self.prototypes.T.to(self.device))  # [B, num_classes]
            
            for i in range(len(batch['seq_ids'])):
                results.append({
                    'seq_id': batch['seq_ids'][i],
                    'true_labels': batch['labels'][i],
                    'logits': logits[i].cpu().numpy()
                })
        
        return results


def extract_logit_true_and_max_incorrect(results, ec_list):
    """
    从伪Logits中提取关键数值：
        - Logit_True: 真实类别的得分
        - Logit_Max_Incorrect: 最大错误类别的得分
    
    参数：
        results: PrototypeBasedEvaluator.extract_logits_for_queries() 的输出
        ec_list: EC号列表（与logits索引对应）
    
    返回：
        logit_true_list: List[float]
        logit_max_incorrect_list: List[float]
    """
    ec_to_idx = {ec: i for i, ec in enumerate(ec_list)}
    
    logit_true_list = []
    logit_max_incorrect_list = []
    
    for record in results:
        true_labels = set(record['true_labels'])
        logits = record['logits']  # [num_classes]
        
        # 提取真实类别的logit（如果有多个，取平均）
        true_indices = [ec_to_idx[ec] for ec in true_labels if ec in ec_to_idx]
        
        if not true_indices:
            continue  # 跳过没有有效标签的样本
        
        logit_true = np.mean([logits[idx] for idx in true_indices])
        
        # 提取最大错误类别的logit
        incorrect_indices = [i for i in range(len(logits)) if i not in true_indices]
        if incorrect_indices:
            logit_max_incorrect = np.max([logits[idx] for idx in incorrect_indices])
        else:
            # 如果所有类别都是正确的（极少见），跳过
            continue
        
        logit_true_list.append(logit_true)
        logit_max_incorrect_list.append(logit_max_incorrect)
    
    return logit_true_list, logit_max_incorrect_list


def visualize_decision_boundary(data_dict, save_path):
    """
    绘制决策边界对比图（小提琴图）
    
    参数：
        data_dict: {
            'Model_Name': {'logit_true': [...], 'logit_max_incorrect': [...]},
            ...
        }
        save_path: 保存路径
    """
    # 准备数据用于绘图
    plot_data = []
    
    for model_name, logits_data in data_dict.items():
        if not logits_data['logit_true']: # 如果列表为空，跳过
            logging.warning(f"No valid data for model {model_name}. Skipping from plot.")
            continue
            
        for logit_type, values in logits_data.items():
            label = "True Class" if logit_type == 'logit_true' else "Max Incorrect"
            for val in values:
                plot_data.append({
                    'Model': model_name,
                    'Type': label,
                    'Logit Score': val
                })
    
    if not plot_data:
        logging.error("No data to plot!")
        return
        
    df = pd.DataFrame(plot_data)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 使用seaborn的violinplot
    sns.violinplot(
        data=df,
        x='Model',
        y='Logit Score',
        hue='Type',
        split=False,  # 分成两个独立的小提琴，更清晰
        inner='box',
        palette={'True Class': '#2ecc71', 'Max Incorrect': '#e74c3c'},
        ax=ax
    )
    
    # 美化
    ax.set_title('Decision Boundary Clarity: Logit Distribution Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Logit Score (Cosine Similarity)', fontsize=14, fontweight='bold')
    ax.legend(title='Logit Type', fontsize=12, title_fontsize=13, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # --- [!!] 已修改的标签逻辑 [!!] ---
    # 我们只展示对您有利的、证明您观点的指标
    
    # 获取 x 轴上的模型顺序
    model_order = [tick.get_text() for tick in ax.get_xticklabels()]

    for i, model_name in enumerate(model_order):
        if model_name not in data_dict or not data_dict[model_name]['logit_max_incorrect']:
            continue
            
        logits_data = data_dict[model_name]
        
        # [!!] 已修改：提取“红色”中位数
        incorrect_median = np.median(logits_data['logit_max_incorrect'])
        
        # 在对应位置添加文本
        y_max_lim = ax.get_ylim()[1]
        y_pos = y_max_lim * 0.95 # 统一放在图表顶部
        
        # [!!] 已修改：更新标签为 "Interference Median" (干扰项中位数)
        # 这个数字越低越好
        ax.text(i, y_pos, f'Interference Median: {incorrect_median:.4f}', 
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Visualization saved to: {save_path}")


def print_statistics(data_dict):
    """打印详细的统计信息"""
    print("\n" + "="*80)
    print("DECISION BOUNDARY STATISTICS")
    print("="*80)
    
    for model_name, logits_data in data_dict.items():
        print(f"\n[{model_name}]")
        print("-" * 40)
        
        if not logits_data['logit_true']:
            print("  No valid data found for this model.")
            continue
            
        true_scores = np.array(logits_data['logit_true'])
        incorrect_scores = np.array(logits_data['logit_max_incorrect'])
        
        # --- [!!] 已修改的日志输出 [!!] ---
        # 1. 突出显示中位数
        # 2. 添加 Median Gap
        
        print(f"Logit_True (Correct Class):")
        print(f"  Mean:   {np.mean(true_scores):.4f}")
        print(f"  Median: {np.median(true_scores):.4f}  <--- (正例重心)")
        print(f"  Std:    {np.std(true_scores):.4f}")
        print(f"  Min:    {np.min(true_scores):.4f}")
        print(f"  Max:    {np.max(true_scores):.4f}")
        
        print(f"\nLogit_Max_Incorrect (Interference):")
        print(f"  Mean:   {np.mean(incorrect_scores):.4f}")
        print(f"  Median: {np.median(incorrect_scores):.4f}  <--- (负例重心，越低越好)")
        print(f"  Std:    {np.std(incorrect_scores):.4f}")
        print(f"  Min:    {np.min(incorrect_scores):.4f}")
        print(f"  Max:    {np.max(incorrect_scores):.4f}")
        
        # 计算各种 Gap
        mean_gap = np.mean(true_scores) - np.mean(incorrect_scores)
        median_gap = np.median(true_scores) - np.median(incorrect_scores)
        overlap = np.sum(incorrect_scores > true_scores) / len(true_scores) * 100
        
        # 计算Effect Size (Cohen's d)
        pooled_std = np.sqrt((np.var(true_scores, ddof=1) + np.var(incorrect_scores, ddof=1)) / 2)
        cohens_d_mean = mean_gap / pooled_std if pooled_std > 0 else 0
        
        print(f"\n{'='*40}")
        print(f"Decision Gap (Mean Diff):   {mean_gap:.4f}  (易受极端值影响)")
        print(f"Decision Gap (Median Diff): {median_gap:.4f}  (更稳健的指标)")
        print(f"Overlap Rate:               {overlap:.2f}%")
        print(f"Effect Size (Cohen's d):    {cohens_d_mean:.4f}")

def load_model_from_checkpoint(checkpoint_path, device):
    """
    从checkpoint加载模型（兼容多种格式）
    
    参数：
        checkpoint_path: checkpoint文件路径
        device: torch.device
    
    返回：
        model: 加载好权重的模型
    """
    model = EcClassifier(**cfg.MODEL_CONFIG).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 兼容两种checkpoint格式
    if 'model_state_dict' in checkpoint:
        # 格式1: {'model_state_dict': {...}, 'optimizer_state_dict': {...}, ...}
        state_dict = checkpoint['model_state_dict']
        logging.info(f"Loaded checkpoint (dictionary format)")
    elif isinstance(checkpoint, dict) and 'fc1.weight' in checkpoint:
        # 格式2: 直接是state_dict
        state_dict = checkpoint
        logging.info(f"Loaded checkpoint (direct state_dict format)")
    else:
        # 兼容一种常见的只保存模型权重的格式
        try:
            model.load_state_dict(checkpoint)
            logging.info(f"Loaded checkpoint (direct state_dict format)")
            return model
        except Exception as e:
            raise ValueError(f"Unknown checkpoint format. Keys found: {checkpoint.keys()}. Error: {e}")
    
    model.load_state_dict(state_dict)
    logging.info(f"Checkpoint loaded from: {checkpoint_path}")
    
    return model


def main():
    # ==========================================================
    # ⚙️ 配置区域 - 请您仔细核对这里的路径
    # ==========================================================
    
    MODEL_CONFIGS = {
        'ESM-2': {
            'checkpoint_path': '/home/fangchh/workdir/triplet/src/experiments/train_final_2025-11-20_03-14-15/best_model.pth',
            'embedding_dir': '/home/fangchh/workdir/triplet/data/ESM_2/protein_embeddings'
        },
        'ESM_CPT': {
            'checkpoint_path': '/home/fangchh/workdir/triplet/src/experiments/train_final_2025-11-19_12-41-24/best_model.pth',
            'embedding_dir': '/home/fangchh/workdir/triplet/data/ESM_CPT/protein_embeddings'
        },
        'ESM_SFT': {
            'checkpoint_path': '/home/fangchh/workdir/triplet/src/experiments/sub/best_model.pth',
            'embedding_dir': '/home/fangchh/workdir/triplet/data/ESM_SUB/protein_embeddings'
        },
        'ESM_ENZ': {
            'checkpoint_path': '/home/fangchh/workdir/triplet/src/experiments/ESM_ENZ/best_model.pth',
            'embedding_dir': '/home/fangchh/workdir/triplet/data/ESM_CPT_SUB/protein_embeddings'
        }
    }
    
    OUTPUT_DIR = 'decision_boundary_experiment'
    
    # ==========================================================
    #  实验开始
    # ==========================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'experiment.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("Starting Decision Boundary Visualization Experiment (v3 - Corrected Paths & Labels)")
    logging.info("="*80)
    logging.info(f"Device: {cfg.DEVICE}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info(f"Models to evaluate: {list(MODEL_CONFIGS.keys())}")
    
    # ==========================================================
    # 📊 加载 Price-149 数据集 (仅加载 ID 和 标签)
    # ==========================================================
    logging.info("\n" + "="*80)
    logging.info("Loading Price-149 dataset (IDs and Labels)...")
    logging.info("="*80)
    
    price_path = cfg.DATA_PATHS["external_test_sets"]["new"]
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Price dataset not found: {price_path}")
    
    # 加载标签映射
    id_to_ecs_map, price_ids = data_utils.load_and_create_label_map(price_path)
    logging.info(f"Loaded {len(price_ids)} Price-149 sample IDs")
    
    # ==========================================================
    #  加载 Gallery（训练集） (仅加载 ID 和 标签)
    # ==========================================================
    logging.info("\n" + "="*80)
    logging.info("Loading Gallery (training set IDs and Labels)...")
    logging.info("="*80)
    
    train_csv = cfg.DATA_PATHS.get("csv_path")
    if not train_csv or not os.path.exists(train_csv):
        logging.warning(f"Train CSV not found: {train_csv}")
        logging.warning("Falling back to using split70.csv")
        train_csv = cfg.DATA_PATHS["csv_path"]
    
    _, train_ids = data_utils.load_and_create_label_map(train_csv)
    train_ids_orig = [entry for entry in train_ids if '_' not in entry]
    
    gallery_ids = train_ids_orig
    
    # 更新id_to_ecs_map以包含训练集的标签
    train_id_to_ecs, _ = data_utils.load_and_create_label_map(train_csv)
    id_to_ecs_map.update(train_id_to_ecs)
    
    logging.info(f"Gallery ID list created: {len(gallery_ids)} samples")

    # ==========================================================
    # 🧠 对每个模型进行实验 (循环已修改)
    # ==========================================================
    results_dict = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        
        checkpoint_path = config['checkpoint_path']
        embedding_dir = config['embedding_dir']
        
        logging.info("\n" + "="*80)
        logging.info(f"Processing Model: {model_name}")
        logging.info(f"  - Checkpoint: {checkpoint_path}")
        logging.info(f"  - Embeddings: {embedding_dir}")
        logging.info("="*80)
        
        if not os.path.exists(embedding_dir):
            logging.warning(f"Embedding directory not found: {embedding_dir}. Skipping model {model_name}.")
            continue
            
        try:
            # --- [!!] 新：在循环内部创建 DataLoaders ---
            
            # 1. 创建 Price-149 (Query) 加载器
            price_dataset = EvaluationDataset(
                price_ids, 
                id_to_ecs_map, 
                embedding_dir,  # <--- 使用了正确的 embedding 路径
                strict_embeddings=False
            )
            price_loader = DataLoader(
                price_dataset, batch_size=32, shuffle=False,
                collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG
            )
            logging.info(f"Price-149 dataset created: {len(price_dataset)} valid samples")

            # 2. 创建 Gallery (原型库) 加载器
            gallery_dataset = EvaluationDataset(
                gallery_ids,
                id_to_ecs_map,
                embedding_dir,  # <--- 使用了正确的 embedding 路径
                strict_embeddings=False # 允许跳过训练集中没有嵌入的
            )
            gallery_loader = DataLoader(
                gallery_dataset, batch_size=32, shuffle=False,
                collate_fn=evaluation_collate_fn, **cfg.DATALOADER_CONFIG
            )
            logging.info(f"Gallery created: {len(gallery_dataset)} valid samples")
            
            if len(price_dataset) == 0 or len(gallery_dataset) == 0:
                logging.warning(f"Skipping {model_name} due to empty dataset (no valid embeddings found).")
                continue

            # --- 评估流程 (与之前相同) ---
            
            # 加载模型
            model = load_model_from_checkpoint(checkpoint_path, cfg.DEVICE)
            
            # 创建评估器（构建原型）
            evaluator = PrototypeBasedEvaluator(model, gallery_loader, cfg.DEVICE)
            
            # 提取Price-149的伪Logits
            results = evaluator.extract_logits_for_queries(price_loader)
            
            # 提取关键数值
            logit_true, logit_max_incorrect = extract_logit_true_and_max_incorrect(
                results, 
                evaluator.ec_list
            )
            
            results_dict[model_name] = {
                'logit_true': logit_true,
                'logit_max_incorrect': logit_max_incorrect
            }
            
            logging.info(f"Successfully extracted {len(logit_true)} valid samples for {model_name}")
            
        except Exception as e:
            logging.error(f"Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # ==========================================================
    # 📈 生成可视化和统计报告
    # ==========================================================
    
    if not results_dict:
        logging.error("No models were successfully evaluated!")
        return
    
    logging.info("\n" + "="*80)
    logging.info("Generating Results...")
    logging.info("="*80)
    
    # 打印统计信息
    print_statistics(results_dict)
    
    # 生成可视化
    viz_path = os.path.join(OUTPUT_DIR, 'decision_boundary_comparison.png')
    visualize_decision_boundary(results_dict, viz_path)
    
    # 保存原始数据
    for model_name, data in results_dict.items():
        csv_path = os.path.join(OUTPUT_DIR, f'{model_name.replace("-", "_").replace(" ", "_")}_logits.csv')
        df = pd.DataFrame({
            'Logit_True': data.get('logit_true', []),
            'Logit_Max_Incorrect': data.get('logit_max_incorrect', [])
        })
        # 仅在数据有效时才添加 Gap
        if 'logit_true' in data and 'logit_max_incorrect' in data and \
           len(data['logit_true']) == len(data['logit_max_incorrect']) and len(data['logit_true']) > 0:
            df['Gap'] = np.array(data['logit_true']) - np.array(data['logit_max_incorrect'])
        
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved raw data to: {csv_path}")
    
    # ==========================================================
    # ✅ 实验完成
    # ==========================================================
    
    logging.info("\n" + "="*80)
    logging.info("🎉 Experiment Completed Successfully!")
    logging.info("="*80)
    logging.info(f"Results saved to: {OUTPUT_DIR}/")
    logging.info(f"  - Visualization: decision_boundary_comparison.png")
    logging.info(f"  - Log file: experiment.log")
    for model_name in results_dict.keys():
        logging.info(f"  - Raw data: {model_name.replace('-', '_').replace(' ', '_')}_logits.csv")
    logging.info("="*80)


if __name__ == '__main__':
    main()