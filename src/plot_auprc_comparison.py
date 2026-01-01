"""
AUPRC Comparison Plot for Multiple ESM Models
==============================================
统一配色和尺寸，适合论文中三张并列展示。

Usage:
    python plot_auprc_comparison_unified.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import Dict, Tuple, List
import logging

# ============================================================================
# 导入项目模块
# ============================================================================
import data_utils
from model import EcClassifier
from datasets import EvaluationDataset
from evaluation import Evaluator, evaluation_collate_fn


# ============================================================================
# 统一配色方案（与回归任务一致）
# ============================================================================
UNIFIED_COLORS = {
    "ESM-2": "#4575b4",       # 深蓝
    "ESM-ENZ": "#d73027",     # 砖红
    "ESM-CPT": "#fc8d59",     # 橙色
    "ESM-SUB": "#91cf60",     # 浅绿
    "ESM-CPT-SUB": "#d73027"  # 砖红（与ESM-ENZ相同）
}

# 统一线型
UNIFIED_LINESTYLES = {
    "ESM-2": "-",
    "ESM-ENZ": "-",
    "ESM-CPT": "--",
    "ESM-SUB": "-.",
    "ESM-CPT-SUB": "-"
}

# ============================================================================
# 配置区
# ============================================================================

# 设备配置
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# 模型配置（与训练时完全一致）
MODEL_CONFIG = {
    'input_dim': 1280,
    'hidden_dim': 512,
    'output_dim': 128,
    'dropout_rate': 0.0,
}

# 评估配置
EVAL_CONFIG = {
    "weighted": True,
    "weighting_mode": "softmax",
    "normalize_embeddings": True,
    "keep_gallery_on_device": True,
    "sim_dtype_fp32": True,
    "default_k": 3,
    "default_tau": 0.35,
    "adaptive_tau": False,
}

K = 3
TAU = 0.35

# DataLoader配置
DATALOADER_CONFIG = {
    "num_workers": 16,
    "pin_memory": True,
    "persistent_workers": True if torch.cuda.is_available() else False,
    "prefetch_factor": 2,
}

BATCH_SIZE = 256

# 数据路径
DATA_PATHS = {
    "train_csv": "/home/fangchh/workdir/triplet/data/split100.csv",
    "price_csv": "/home/fangchh/workdir/triplet/data/price.csv",
}

# 模型配置
MODELS_CONFIG = {
    "ESM-2": (
        "/home/fangchh/workdir/triplet/data/ESM_2/protein_embeddings",
        "experiments/esm/best_model.pth"
    ),
    "ESM-ENZ": (
        "/home/fangchh/workdir/triplet/data/ESM_CPT_SUB/protein_embeddings",
        "experiments/cpt_sub/best_model.pth"
    ),
}

# 输出配置
OUTPUT_DIR = "figures"
OUTPUT_FILENAME = "auprc_comparison_price"
OUTPUT_TITLE = "EC Classification"  # 默认标题


# ============================================================================
# 日志配置
# ============================================================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


# ============================================================================
# 核心函数
# ============================================================================

def prepare_data_for_model(
    embedding_dir: str,
    train_csv: str,
    test_csv: str
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """为单个模型准备数据"""
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(train_csv)
    id_to_ecs_test, test_ids_raw = data_utils.load_and_create_label_map(test_csv)
    id_to_ecs_master.update(id_to_ecs_test)
    
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    
    id_to_ecs_master = data_utils.augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=embedding_dir,
        train_ids_orig_only=all_original_ids
    )
    
    train_set = set(all_original_ids)
    mutants_train = [
        sid for sid in id_to_ecs_master.keys() 
        if '_' in sid and sid.split('_')[0] in train_set
    ]
    
    gallery_ids_raw = all_original_ids + mutants_train
    
    gallery_ids = data_utils.filter_ids_with_embeddings(
        gallery_ids_raw, embedding_dir, "[Gallery] "
    )
    test_ids = data_utils.filter_ids_with_embeddings(
        test_ids_raw, embedding_dir, "[Test] "
    )
    
    return gallery_ids, test_ids, id_to_ecs_master


def evaluate_model_and_get_pr_data(
    model_name: str,
    embedding_dir: str,
    model_path: str,
    gallery_ids: List[str],
    test_ids: List[str],
    id_to_ecs_master: Dict[str, List[str]],
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, float]:
    """评估单个模型并返回PR曲线数据"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluating: {model_name}")
    logging.info(f"{'='*60}")
    
    gallery_dataset = EvaluationDataset(
        gallery_ids, id_to_ecs_master, embedding_dir
    )
    test_dataset = EvaluationDataset(
        test_ids, id_to_ecs_master, embedding_dir, strict_embeddings=False
    )
    
    logging.info(f"Gallery size: {len(gallery_dataset)}")
    logging.info(f"Test size: {len(test_dataset)}")
    
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=evaluation_collate_fn, **DATALOADER_CONFIG
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=evaluation_collate_fn, **DATALOADER_CONFIG
    )
    
    model = EcClassifier(**MODEL_CONFIG).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    evaluator = Evaluator(
        model=model,
        gallery_loader=gallery_loader,
        device=device,
        eval_config=EVAL_CONFIG
    )
    
    eval_output = evaluator.evaluate(
        query_loader=test_loader,
        k=K,
        vote_tau=TAU,
        query_set_name=f"Test_{model_name}",
        save_preds=False,
        plot_auprc=False,
        return_raw_scores=True
    )
    
    metrics = {k: v for k, v in eval_output.items() if k != 'raw_data'}
    logging.info(f"Metrics: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, "
                 f"Recall={metrics['recall']:.4f}, AUC={metrics['auc']:.4f}")
    
    if 'raw_data' not in eval_output:
        raise ValueError(f"No raw_data returned for {model_name}")
    
    y_true, y_score = eval_output['raw_data']
    auprc = average_precision_score(y_true, y_score)
    logging.info(f"AUPRC: {auprc:.4f}")
    
    return y_true, y_score, auprc


def plot_pr_curve_single(
    pr_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: str,
    title: str = None,
    figsize: Tuple[float, float] = (3.3, 3.3)
):
    """
    绘制单个PR曲线图（适合三张并列）
    
    Args:
        pr_data_dict: {模型名: (y_true, y_score, auprc)}
        output_path: 输出路径
        title: 图标题
        figsize: 图尺寸，默认(3.3, 3.3)适合三张并列
    """
    # Nature/Science 风格设置
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.linewidth': 1.0,
        'axes.labelsize': 10,
        'axes.titlesize': 6,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 按AUPRC排序绘制
    sorted_models = sorted(pr_data_dict.items(), key=lambda x: -x[1][2])
    
    for model_name, (y_true, y_score, auprc) in sorted_models:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        color = UNIFIED_COLORS.get(model_name, '#333333')
        ls = UNIFIED_LINESTYLES.get(model_name, '-')
        
        ax.plot(
            recall, precision, 
            color=color, 
            linestyle=ls,
            linewidth=1.8,
            label=f'{model_name} ({auprc:.3f})'
        )
    
    # 随机基线
    first_y_true = list(pr_data_dict.values())[0][0]
    baseline = np.sum(first_y_true) / len(first_y_true)
    ax.axhline(
        y=baseline, 
        color='#888888', 
        linestyle=':', 
        linewidth=1.0,
        label=f'Random ({baseline:.3f})',
        zorder=0
    )
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    
    # 图例
    legend = ax.legend(
        loc='lower left',
        framealpha=1.0,
    )
    legend.get_frame().set_linewidth(0.6)
    
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    
    plt.tight_layout()
    
    # 保存
    for ext in ['pdf', 'png']:
        save_path = output_path.replace('.png', f'.{ext}').replace('.pdf', f'.{ext}')
        if not save_path.endswith(f'.{ext}'):
            save_path = f"{output_path}.{ext}"
        dpi = 600 if ext == 'png' else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        logging.info(f"Saved: {save_path}")
    
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def plot_combined_pr_curves(
    pr_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_dir: str,
    output_filename: str,
    title: str = None
):
    """绘制多模型PR曲线对比图（兼容旧接口）"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plot_pr_curve_single(pr_data_dict, output_path, title=title, figsize=(3.3, 3.3))


# ============================================================================
# 独立绘图函数（直接从CSV读取预测结果）
# ============================================================================

def plot_pr_from_predictions(
    esm2_csv: str,
    enz_csv: str,
    output_path: str,
    title: str = None,
    true_col: str = 'y_true',
    score_col: str = 'y_score',
    figsize: Tuple[float, float] = (3.3, 3.3)
):
    """
    从predictions CSV文件绘制PR曲线
    
    Args:
        esm2_csv: ESM-2 预测结果CSV路径
        enz_csv: ESM-ENZ 预测结果CSV路径
        output_path: 输出路径
        title: 图标题
        true_col: 真实标签列名
        score_col: 预测分数列名
        figsize: 图尺寸
    """
    import pandas as pd
    
    pr_data_dict = {}
    
    # 加载ESM-2
    if os.path.exists(esm2_csv):
        df = pd.read_csv(esm2_csv)
        y_true = df[true_col].values
        y_score = df[score_col].values
        auprc = average_precision_score(y_true, y_score)
        pr_data_dict['ESM-2'] = (y_true, y_score, auprc)
        logging.info(f"ESM-2 AUPRC: {auprc:.4f}")
    
    # 加载ESM-ENZ
    if os.path.exists(enz_csv):
        df = pd.read_csv(enz_csv)
        y_true = df[true_col].values
        y_score = df[score_col].values
        auprc = average_precision_score(y_true, y_score)
        pr_data_dict['ESM-ENZ'] = (y_true, y_score, auprc)
        logging.info(f"ESM-ENZ AUPRC: {auprc:.4f}")
    
    if pr_data_dict:
        plot_pr_curve_single(pr_data_dict, output_path, title, figsize)
    else:
        logging.error("No valid data found!")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AUPRC Comparison Plot (Unified Style)")
    parser.add_argument("--title", type=str, default=OUTPUT_TITLE,
                       help="图标题，如 'EC Classification'")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                       help="输出目录")
    parser.add_argument("--output-filename", type=str, default=OUTPUT_FILENAME,
                       help="输出文件名（不含扩展名）")
    parser.add_argument("--device", type=str, default="cuda:2",
                       help="计算设备")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    setup_logging()
    
    logging.info("="*60)
    logging.info("AUPRC Comparison Plot Generator (Unified Style)")
    logging.info("="*60)
    logging.info(f"Device: {device}")
    logging.info(f"Title: {args.title}")
    logging.info(f"Figure size: 3.3 x 3.3 inches (for 3-panel layout)")
    
    pr_data_dict = {}
    
    for model_name, (embedding_dir, model_path) in MODELS_CONFIG.items():
        try:
            logging.info(f"\nPreparing data for {model_name}...")
            gallery_ids, test_ids, id_to_ecs_master = prepare_data_for_model(
                embedding_dir=embedding_dir,
                train_csv=DATA_PATHS["train_csv"],
                test_csv=DATA_PATHS["price_csv"]
            )
            
            y_true, y_score, auprc = evaluate_model_and_get_pr_data(
                model_name=model_name,
                embedding_dir=embedding_dir,
                model_path=model_path,
                gallery_ids=gallery_ids,
                test_ids=test_ids,
                id_to_ecs_master=id_to_ecs_master,
                device=device
            )
            
            pr_data_dict[model_name] = (y_true, y_score, auprc)
            
        except Exception as e:
            logging.error(f"Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not pr_data_dict:
        logging.error("No models were successfully evaluated!")
        return
    
    logging.info("\n" + "="*60)
    logging.info("Generating comparison plot...")
    logging.info("="*60)
    
    plot_combined_pr_curves(
        pr_data_dict=pr_data_dict,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        title=args.title
    )
    
    logging.info("\n" + "="*60)
    logging.info("AUPRC Summary")
    logging.info("="*60)
    for model_name, (_, _, auprc) in sorted(pr_data_dict.items(), key=lambda x: -x[1][2]):
        logging.info(f"  {model_name}: {auprc:.4f}")
    
    logging.info("\nDone!")


if __name__ == "__main__":
    main()