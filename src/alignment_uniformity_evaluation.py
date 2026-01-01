"""
alignment_uniformity_evaluation.py
完整的Alignment & Uniformity评估框架

使用方法：
    python alignment_uniformity_evaluation.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import logging
from scipy import stats

# 导入你的项目模块
import config as cfg
from model import EcClassifier
from datasets import EvaluationDataset
from evaluation import evaluation_collate_fn


class AlignmentUniformityEvaluator:
    """
    Alignment & Uniformity评估器
    
    理论基础：
    Wang & Isola, "Understanding Contrastive Representation Learning 
    through Alignment and Uniformity on the Hypersphere", ICML 2020
    """
    
    def __init__(self, temperature=0.1, uniformity_t=2):
        """
        Args:
            temperature: Alignment计算的温度参数（不影响排序，但影响数值）
            uniformity_t: Uniformity的温度参数（论文推荐t=2）
        """
        self.temperature = temperature
        self.uniformity_t = uniformity_t
    
    @torch.no_grad()
    def compute_alignment(self, embeddings, labels, use_sampling=True, max_pairs=50000):
        """
        计算Alignment：同类样本之间的平均距离
        
        L_align = E_{(x,x+)~p_pos} [||f(x) - f(x+)||²]
        
        Args:
            embeddings: [N, D] tensor，已归一化
            labels: List[List[str]]，每个样本的EC标签列表
            use_sampling: 是否采样（数据量大时）
            max_pairs: 最大采样对数
        
        Returns:
            alignment_loss: float，越小越好
            alignment_dict: 详细统计信息
        """
        
        logging.info("Computing Alignment (intra-class compactness)...")
        
        # 构建EC到样本索引的映射
        ec_to_indices = defaultdict(list)
        for i, label_list in enumerate(labels):
            for ec in label_list:
                ec_to_indices[ec].append(i)
        
        # 收集所有正样本对的距离
        positive_distances = []
        pair_counts = defaultdict(int)  # 统计每个EC的样本对数量
        
        for ec, indices in tqdm(ec_to_indices.items(), desc="Computing positive pairs"):
            if len(indices) < 2:
                continue  # 该EC只有一个样本，跳过
            
            # 获取该EC的所有样本嵌入
            ec_embeddings = embeddings[indices]  # [n, D]
            
            # 计算所有样本对的L2距离平方
            # distances[i,j] = ||emb[i] - emb[j]||²
            distances = torch.cdist(ec_embeddings, ec_embeddings, p=2).pow(2)
            
            # 取上三角（避免重复和自身）
            upper_tri_indices = torch.triu_indices(len(indices), len(indices), offset=1)
            pair_distances = distances[upper_tri_indices[0], upper_tri_indices[1]]
            
            # 采样（如果样本对太多）
            if use_sampling and len(pair_distances) > max_pairs // len(ec_to_indices):
                sample_size = min(len(pair_distances), max_pairs // len(ec_to_indices))
                sampled_indices = torch.randperm(len(pair_distances))[:sample_size]
                pair_distances = pair_distances[sampled_indices]
            
            positive_distances.extend(pair_distances.cpu().numpy())
            pair_counts[ec] = len(pair_distances)
        
        if not positive_distances:
            logging.warning("No positive pairs found!")
            return float('inf'), {}
        
        positive_distances = np.array(positive_distances)
        
        # 计算Alignment loss（平均距离）
        alignment_loss = float(positive_distances.mean())
        
        # 详细统计
        alignment_dict = {
            'alignment_loss': alignment_loss,
            'num_positive_pairs': len(positive_distances),
            'num_ecs_with_pairs': len(pair_counts),
            'mean_distance': float(positive_distances.mean()),
            'median_distance': float(np.median(positive_distances)),
            'std_distance': float(positive_distances.std()),
            'min_distance': float(positive_distances.min()),
            'max_distance': float(positive_distances.max()),
            'percentile_25': float(np.percentile(positive_distances, 25)),
            'percentile_75': float(np.percentile(positive_distances, 75)),
        }
        
        logging.info(f"  ✓ Alignment Loss: {alignment_loss:.6f}")
        logging.info(f"  ✓ Positive Pairs: {len(positive_distances):,}")
        
        return alignment_loss, alignment_dict
    
    @torch.no_grad()
    def compute_uniformity(self, embeddings, use_sampling=True, max_samples=2000):
        """
        计算Uniformity：样本在超球面上的分布均匀性
        
        L_uniform = log E_{x,y~p_data} [e^(-t||f(x)-f(y)||²)]
        
        Args:
            embeddings: [N, D] tensor，已归一化
            use_sampling: 是否采样（数据量大时）
            max_samples: 最大采样样本数
        
        Returns:
            uniformity_loss: float，越小越好
            uniformity_dict: 详细统计信息
        """
        
        logging.info("Computing Uniformity (hypersphere coverage)...")
        
        n = len(embeddings)
        
        # 如果样本太多，随机采样
        if use_sampling and n > max_samples:
            indices = torch.randperm(n)[:max_samples]
            embeddings = embeddings[indices]
            n = max_samples
            logging.info(f"  Sampled {n} embeddings for uniformity computation")
        
        # 计算所有样本对的L2距离平方
        # 注意：这里使用的是完整的距离矩阵（包括对角线）
        distances = torch.cdist(embeddings, embeddings, p=2).pow(2)  # [n, n]
        
        # 计算uniformity loss
        # log E[e^(-t*d²)] = logsumexp(-t*d²) - log(n²)
        uniformity_loss = torch.logsumexp(-self.uniformity_t * distances, dim=(0, 1)).item()
        uniformity_loss = uniformity_loss - np.log(n * n)
        
        # 额外统计：样本之间的平均距离
        # 取上三角（排除对角线和重复）
        upper_tri_indices = torch.triu_indices(n, n, offset=1)
        pairwise_distances = distances[upper_tri_indices[0], upper_tri_indices[1]].cpu().numpy()
        
        uniformity_dict = {
            'uniformity_loss': uniformity_loss,
            'num_samples': n,
            'mean_pairwise_distance': float(pairwise_distances.mean()),
            'std_pairwise_distance': float(pairwise_distances.std()),
            'min_pairwise_distance': float(pairwise_distances.min()),
            'max_pairwise_distance': float(pairwise_distances.max()),
        }
        
        logging.info(f"  ✓ Uniformity Loss: {uniformity_loss:.6f}")
        logging.info(f"  ✓ Samples Used: {n:,}")
        
        return uniformity_loss, uniformity_dict
    
    def evaluate(self, model, data_loader, device, dataset_name='Test'):
        """
        完整评估流程
        
        Returns:
            results: dict，包含所有指标
        """
        
        logging.info(f"\n{'='*80}")
        logging.info(f" Evaluating {dataset_name} Set ")
        logging.info(f"{'='*80}")
        
        model.eval()
        
        # 收集所有embeddings和labels
        all_embeddings = []
        all_labels = []
        
        logging.info("Extracting embeddings...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches"):
                if batch.get("empty", False):
                    continue
                
                embeddings_1280 = batch['embedding_1280'].to(device)
                embeddings_128 = model(embeddings_1280)  # [B, 128] 已归一化
                
                all_embeddings.append(embeddings_128.cpu())
                all_labels.extend(batch['labels'])
        
        if not all_embeddings:
            logging.error("No valid embeddings extracted!")
            return None
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        logging.info(f"Total samples: {len(all_embeddings)}")
        logging.info(f"Embedding dimension: {all_embeddings.shape[1]}")
        
        # 验证归一化
        norms = torch.norm(all_embeddings, p=2, dim=1)
        logging.info(f"Embedding norms - mean: {norms.mean():.6f}, std: {norms.std():.6f}")
        
        # 计算Alignment
        alignment_loss, alignment_dict = self.compute_alignment(
            all_embeddings, all_labels
        )
        
        # 计算Uniformity
        uniformity_loss, uniformity_dict = self.compute_uniformity(
            all_embeddings
        )
        
        # 综合结果
        results = {
            'dataset_name': dataset_name,
            'num_samples': len(all_embeddings),
            'alignment': alignment_dict,
            'uniformity': uniformity_dict,
            'alignment_loss': alignment_loss,
            'uniformity_loss': uniformity_loss,
        }
        
        # 打印摘要
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """打印评估摘要"""
        
        print("\n" + "="*80)
        print(f" {results['dataset_name']} - Alignment & Uniformity Summary ")
        print("="*80)
        
        print(f"\n📊 Alignment (Intra-class Compactness):")
        print(f"  Loss:           {results['alignment_loss']:.6f}  ← Lower is better")
        print(f"  Mean Distance:  {results['alignment']['mean_distance']:.6f}")
        print(f"  Median Distance:{results['alignment']['median_distance']:.6f}")
        print(f"  Positive Pairs: {results['alignment']['num_positive_pairs']:,}")
        
        print(f"\n📊 Uniformity (Hypersphere Coverage):")
        print(f"  Loss:           {results['uniformity_loss']:.6f}  ← Lower is better")
        print(f"  Mean Pairwise:  {results['uniformity']['mean_pairwise_distance']:.6f}")
        
        print("\n" + "="*80)


def compare_models(results_dict, save_dir):
    """
    对比多个模型的Alignment & Uniformity
    
    Args:
        results_dict: {
            'ESM_2': {'PRICE': results, 'NEW': results},
            'ESM_CPT': {'PRICE': results, 'NEW': results}
        }
        save_dir: 保存路径
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(" MODEL COMPARISON: Alignment & Uniformity ")
    print("="*80)
    
    # 准备数据
    comparison_data = []
    
    for model_name, datasets in results_dict.items():
        for dataset_name, results in datasets.items():
            comparison_data.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Alignment': results['alignment_loss'],
                'Uniformity': results['uniformity_loss'],
                'N_Samples': results['num_samples']
            })
    
    df = pd.DataFrame(comparison_data)
    
    # 1. 打印对比表
    print("\n" + "-"*80)
    print(f"{'Model':<15} {'Dataset':<10} {'Alignment':<15} {'Uniformity':<15}")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{row['Model']:<15} {row['Dataset']:<10} "
              f"{row['Alignment']:<15.6f} {row['Uniformity']:<15.6f}")
    
    # 2. 计算相对改进
    if 'ESM_2' in results_dict and 'ESM_CPT' in results_dict:
        print("\n" + "="*80)
        print(" CPT IMPROVEMENT over ESM-2 ")
        print("="*80)
        
        for dataset in ['PRICE', 'NEW']:
            if dataset in results_dict['ESM_2'] and dataset in results_dict['ESM_CPT']:
                base = results_dict['ESM_2'][dataset]
                cpt = results_dict['ESM_CPT'][dataset]
                
                align_improve = (base['alignment_loss'] - cpt['alignment_loss']) / base['alignment_loss'] * 100
                uniform_improve = (base['uniformity_loss'] - cpt['uniformity_loss']) / base['uniformity_loss'] * 100
                
                print(f"\n{dataset}:")
                print(f"  Alignment improvement:  {align_improve:+.2f}%  "
                      f"({'better' if align_improve > 0 else 'worse'})")
                print(f"  Uniformity improvement: {uniform_improve:+.2f}%  "
                      f"({'better' if uniform_improve > 0 else 'worse'})")
                
                # 统计显著性检验（需要原始距离数据）
                # 这里简化处理，实际可以保存原始距离进行t-test
    
    # 3. 保存CSV
    csv_path = os.path.join(save_dir, 'alignment_uniformity_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison table saved to: {csv_path}")
    
    # 4. 可视化
    plot_comparison(df, save_dir)
    
    return df


def plot_comparison(df, save_dir):
    """
    生成对比可视化
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1：Alignment对比
    ax1 = axes[0]
    
    datasets = df['Dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    models = df['Model'].unique()
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        alignments = [model_data[model_data['Dataset']==ds]['Alignment'].values[0] 
                     for ds in datasets]
        
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax1.bar(x + offset, alignments, width, label=model, alpha=0.8)
        
        # 标注数值
        for j, (bar, val) in enumerate(zip(bars, alignments)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Alignment Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Alignment: Intra-class Compactness\n(Lower is Better)', 
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2：Uniformity对比
    ax2 = axes[1]
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        uniformities = [model_data[model_data['Dataset']==ds]['Uniformity'].values[0] 
                       for ds in datasets]
        
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax2.bar(x + offset, uniformities, width, label=model, alpha=0.8)
        
        for j, (bar, val) in enumerate(zip(bars, uniformities)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Uniformity Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Uniformity: Hypersphere Coverage\n(Lower is Better)', 
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'alignment_uniformity_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {save_path}")


def plot_alignment_uniformity_scatter(results_dict, save_dir):
    """
    绘制Alignment vs Uniformity散点图
    
    理想模型应该位于左下角（低Alignment + 低Uniformity）
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'PRICE': '#e74c3c', 'NEW': '#3498db'}
    markers = {'ESM_2': 'o', 'ESM_CPT': 's', 'ESM_SFT': '^', 'ESM_ENZ': 'D'}
    
    for model_name, datasets in results_dict.items():
        for dataset_name, results in datasets.items():
            ax.scatter(
                results['alignment_loss'],
                results['uniformity_loss'],
                s=200,
                marker=markers.get(model_name, 'o'),
                color=colors.get(dataset_name, 'gray'),
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5,
                label=f'{model_name}-{dataset_name}'
            )
            
            # 标注
            ax.annotate(
                f'{model_name}\n{dataset_name}',
                (results['alignment_loss'], results['uniformity_loss']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
    
    ax.set_xlabel('Alignment Loss (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Uniformity Loss (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_title('Alignment vs Uniformity Trade-off\n(Ideal: Bottom-Left Corner)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # 添加"理想区域"标注
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.annotate(
        '← Ideal Region',
        xy=(xlim[0] + (xlim[1]-xlim[0])*0.1, ylim[0] + (ylim[1]-ylim[0])*0.1),
        fontsize=12,
        fontweight='bold',
        color='green',
        alpha=0.6
    )
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'alignment_vs_uniformity_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Scatter plot saved to: {save_path}")


def load_model_and_data(model_config, test_ids_dict, id_to_ecs_map):
    """
    加载模型和数据
    """
    
    # 加载模型
    model = EcClassifier(**cfg.MODEL_CONFIG).to(cfg.DEVICE)
    checkpoint = torch.load(model_config['checkpoint'], map_location=cfg.DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 创建数据加载器
    data_loaders = {}
    
    for dataset_name, test_ids in test_ids_dict.items():
        dataset = EvaluationDataset(
            test_ids,
            id_to_ecs_map,
            model_config['embedding_dir'],
            strict_embeddings=False
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=evaluation_collate_fn,
            **cfg.DATALOADER_CONFIG
        )
        
        data_loaders[dataset_name] = data_loader
    
    return model, data_loaders


def main():
    """
    主程序
    """
    
    # ============================================================
    # 配置
    # ============================================================
    
    MODEL_CONFIGS = {
        'ESM_2': {
            'checkpoint': '/home/fangchh/workdir/triplet/src/experiments/esm/best_model.pth',
            'embedding_dir': '/home/fangchh/workdir/triplet/data/ESM_2/protein_embeddings'
        },
        'ESM_CPT': {
            'checkpoint': '/home/fangchh/workdir/triplet/src/experiments/cpt/best_model.pth',
            'embedding_dir': '/home/fangchh/workdir/triplet/data/ESM_CPT/protein_embeddings'
        }
    }
    
    OUTPUT_DIR = 'alignment_uniformity_evaluation'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info(" Alignment & Uniformity Evaluation ")
    logging.info("="*80)
    
    # ============================================================
    # 加载测试集ID和标签
    # ============================================================
    
    import data_utils
    
    # PRICE测试集
    price_path = cfg.DATA_PATHS["external_test_sets"]["price"]
    price_id_to_ecs, price_ids = data_utils.load_and_create_label_map(price_path)
    
    # NEW测试集
    new_path = cfg.DATA_PATHS["external_test_sets"]["new"]
    new_id_to_ecs, new_ids = data_utils.load_and_create_label_map(new_path)
    
    # 合并标签映射
    id_to_ecs_map = {**price_id_to_ecs, **new_id_to_ecs}
    
    test_ids_dict = {
        'PRICE': price_ids,
        'NEW': new_ids
    }
    
    logging.info(f"Loaded test sets:")
    logging.info(f"  PRICE: {len(price_ids)} samples")
    logging.info(f"  NEW:   {len(new_ids)} samples")
    
    # ============================================================
    # 评估所有模型
    # ============================================================
    
    evaluator = AlignmentUniformityEvaluator(temperature=0.1, uniformity_t=2)
    
    all_results = {}
    
    for model_name, model_config in MODEL_CONFIGS.items():
        logging.info(f"\n{'='*80}")
        logging.info(f" Processing Model: {model_name} ")
        logging.info(f"{'='*80}")
        
        try:
            # 加载模型和数据
            model, data_loaders = load_model_and_data(
                model_config, test_ids_dict, id_to_ecs_map
            )
            
            # 评估每个测试集
            model_results = {}
            
            for dataset_name, data_loader in data_loaders.items():
                results = evaluator.evaluate(
                    model, data_loader, cfg.DEVICE, dataset_name
                )
                
                if results:
                    model_results[dataset_name] = results
            
            all_results[model_name] = model_results
            
        except Exception as e:
            logging.error(f"Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================================
    # 对比和可视化
    # ============================================================
    
    if len(all_results) >= 2:
        logging.info("\n" + "="*80)
        logging.info(" Generating Comparison Results ")
        logging.info("="*80)
        
        # 生成对比表和图表
        comparison_df = compare_models(all_results, OUTPUT_DIR)
        
        # 生成散点图
        plot_alignment_uniformity_scatter(all_results, OUTPUT_DIR)
        
        # 保存完整结果
        import json
        with open(os.path.join(OUTPUT_DIR, 'full_results.json'), 'w') as f:
            # 转换为可序列化的格式
            serializable_results = {}
            for model_name, datasets in all_results.items():
                serializable_results[model_name] = {}
                for dataset_name, results in datasets.items():
                    serializable_results[model_name][dataset_name] = {
                        k: v for k, v in results.items() 
                        if not isinstance(v, (torch.Tensor, np.ndarray))
                    }
            
            json.dump(serializable_results, f, indent=2)
        
        logging.info(f"\n✓ Full results saved to: {os.path.join(OUTPUT_DIR, 'full_results.json')}")
    
    # ============================================================
    # 完成
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" 🎉 Evaluation Completed Successfully! ")
    logging.info("="*80)
    logging.info(f"\nResults saved to: {OUTPUT_DIR}/")
    logging.info(f"  - alignment_uniformity_comparison.csv")
    logging.info(f"  - alignment_uniformity_comparison.png")
    logging.info(f"  - alignment_vs_uniformity_scatter.png")
    logging.info(f"  - full_results.json")
    logging.info(f"  - evaluation.log")
    logging.info("="*80)


if __name__ == '__main__':
    main()