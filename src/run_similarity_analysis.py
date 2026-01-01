"""
run_similarity_analysis.py
主程序：运行完整的相似度分析实验
"""
import os
import sys
from datetime import datetime
import logging
import torch
from torch.utils.data import DataLoader

# 导入项目模块
import config as cfg
import data_utils
import utils
from model import EcClassifier
from datasets import EvaluationDataset
from evaluation import evaluation_collate_fn

# 导入分析模块
from similarity_analysis import collect_all_models_data
from plot_similarity_analysis import (
    plot_figure1_top1_similarity,
    plot_figure2_margin_comparison,
    plot_figure3_margin_vs_accuracy,
    plot_figure4_cdf_curves,
    plot_figure5_entropy_comparison,
    plot_figure6_gap_ratio_vs_f1,
    generate_summary_table
)


def main():
    """
    主函数：执行完整的相似度分析流程
    """
    
    # ============================================================
    # 1. 初始化
    # ============================================================
    
    # 创建输出目录
    output_dir = os.path.join(
        "experiments", 
        f"similarity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(output_dir, "analysis.log")
    utils.setup_logging(log_path)
    
    logging.info("="*80)
    logging.info(" Similarity Distribution Analysis - Representation Quality Study ")
    logging.info("="*80)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {cfg.DEVICE}")
    
    # ============================================================
    # 2. 配置模型
    # ============================================================
    model_configs = {
        'ESM_2': {
            'path': 'experiments/esm/best_model.pth',
            'model_class': EcClassifier,
            'model_kwargs': cfg.MODEL_CONFIG
        },
        'ESM_CPT': {
            'path': 'experiments/cpt/best_model.pth',
            'model_class': EcClassifier,
            'model_kwargs': cfg.MODEL_CONFIG
        },
        'ESM_SUB': {
            'path': 'experiments/sub/best_model.pth',
            'model_class': EcClassifier,
            'model_kwargs': cfg.MODEL_CONFIG
        },
        'ESM_ENZ': {
            'path': 'experiments/cpt_sub/best_model.pth',
            'model_class': EcClassifier,
            'model_kwargs': cfg.MODEL_CONFIG
        },
    }
    
    logging.info("\nModels to analyze:")
    for model_name, model_cfg in model_configs.items():
        logging.info(f"  - {model_name}: {model_cfg['path']}")
        if not os.path.exists(model_cfg['path']):
            logging.error(f"    ✗ Model file not found!")
            sys.exit(1)
        else:
            logging.info(f"    ✓ Model file exists")
    
    # ============================================================
    # 3. 准备数据
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" Preparing Data ")
    logging.info("="*80)
    
    # 加载标签映射
    logging.info("Loading label mappings...")
    id_to_ecs_master, all_ids = data_utils.load_and_create_label_map(
        cfg.DATA_PATHS["csv_path"]
    )
    
    # 加载外部测试集标签
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        id_to_ecs_ext, _ = data_utils.load_and_create_label_map(path)
        id_to_ecs_master.update(id_to_ecs_ext)
        logging.info(f"  Loaded {name}: {len(id_to_ecs_ext)} samples")
    
    # 获取原始IDs
    all_original_ids = sorted([entry for entry in all_ids if '_' not in entry])
    logging.info(f"Total original IDs: {len(all_original_ids)}")
    
    # 增强训练数据（用于Gallery）
    logging.info("Augmenting with mutants for gallery...")
    id_to_ecs_master = data_utils.augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=cfg.DATA_PATHS["embedding_dir"],
        train_ids_orig_only=all_original_ids
    )
    
    mutants_train = [sid for sid in id_to_ecs_master.keys() 
                    if '_' in sid and sid.split('_')[0] in set(all_original_ids)]
    
    train_pool_ids_raw = all_original_ids + mutants_train
    logging.info(f"Gallery pool size: {len(train_pool_ids_raw)} (originals + mutants)")
    
    # 构建Gallery
    logging.info("Building augmented gallery for retrieval...")
    gallery_ids = data_utils.filter_ids_with_embeddings(
        train_pool_ids_raw, 
        cfg.DATA_PATHS["embedding_dir"], 
        "[Gallery] "
    )
    
    # ============================================================
    # 4. 创建DataLoaders
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" Creating DataLoaders ")
    logging.info("="*80)
    
    batch_size = 64  # 推理时可以用更大的batch
    
    # Gallery DataLoader
    gallery_dataset = EvaluationDataset(
        gallery_ids, 
        id_to_ecs_master, 
        cfg.DATA_PATHS["embedding_dir"]
    )
    gallery_loader = DataLoader(
        gallery_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=evaluation_collate_fn, 
        **cfg.DATALOADER_CONFIG
    )
    logging.info(f"Gallery: {len(gallery_dataset)} samples")
    
    # 测试集DataLoaders
    test_loaders = {}
    for name, path in cfg.DATA_PATHS["external_test_sets"].items():
        _, ext_test_ids = data_utils.load_and_create_label_map(path)
        ext_test_dataset = EvaluationDataset(
            ext_test_ids, 
            id_to_ecs_master, 
            cfg.DATA_PATHS["embedding_dir"], 
            strict_embeddings=False
        )
        test_loaders[name.upper()] = DataLoader(
            ext_test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=evaluation_collate_fn, 
            **cfg.DATALOADER_CONFIG
        )
        logging.info(f"Test set {name.upper()}: {len(ext_test_dataset)} samples")
    
    # ============================================================
    # 5. 收集相似度数据
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 1: Collecting Similarity Metrics ")
    logging.info("="*80)
    
    all_data = collect_all_models_data(
        model_configs=model_configs,
        gallery_loader=gallery_loader,
        test_loaders=test_loaders,
        device=cfg.DEVICE,
        eval_config=cfg.EVAL_CONFIG,
        output_dir=output_dir,
        k=3  # 收集top-3的相似度
    )
    
    if not all_data:
        logging.error("No data collected! Exiting...")
        sys.exit(1)
    
    logging.info("\n✓ Data collection completed!")
    logging.info(f"  Models processed: {len(all_data)}")
    for model_name, test_data in all_data.items():
        logging.info(f"  {model_name}: {list(test_data.keys())}")
    
    # ============================================================
    # 6. 生成可视化图表
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 2: Generating Visualizations ")
    logging.info("="*80)
    
    # 图1: Top-1相似度分布
    plot_figure1_top1_similarity(
        all_data,
        os.path.join(output_dir, "fig1_top1_similarity.png")
    )
    
    # 图2: Margin对比（含显著性检验）
    plot_figure2_margin_comparison(
        all_data,
        os.path.join(output_dir, "fig2_margin_comparison.png")
    )
    
    # 图3: Margin vs Accuracy（最关键）
    plot_figure3_margin_vs_accuracy(
        all_data,
        os.path.join(output_dir, "fig3_margin_vs_accuracy.png")
    )
    
    # 图4: CDF曲线
    plot_figure4_cdf_curves(
        all_data,
        os.path.join(output_dir, "fig4_cdf_curves.png")
    )
    
    # 图5: 熵分布
    plot_figure5_entropy_comparison(
        all_data,
        os.path.join(output_dir, "fig5_entropy_distribution.png")
    )
    
    # 图6: Gap Ratio vs F1
    plot_figure6_gap_ratio_vs_f1(
        all_data,
        os.path.join(output_dir, "fig6_gap_ratio_vs_f1.png")
    )
    
    # ============================================================
    # 7. 生成统计摘要
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" STEP 3: Generating Summary Statistics ")
    logging.info("="*80)
    
    summary_df = generate_summary_table(all_data)
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"✓ Summary table saved to {summary_path}")
    
    # 打印摘要到控制台
    logging.info("\n" + "="*80)
    logging.info(" Summary Statistics ")
    logging.info("="*80)
    print("\n" + summary_df.to_string(index=False))
    
    # ============================================================
    # 8. 完成
    # ============================================================
    
    logging.info("\n" + "="*80)
    logging.info(" Analysis Completed Successfully! ")
    logging.info("="*80)
    logging.info(f"\nAll outputs saved to: {output_dir}")
    logging.info("\nGenerated files:")
    logging.info("  Data:")
    logging.info("    - similarity_metrics_*.csv (raw data for each model-testset)")
    logging.info("    - summary_statistics.csv (aggregated statistics)")
    logging.info("  Figures:")
    logging.info("    - fig1_top1_similarity.png")
    logging.info("    - fig2_margin_comparison.png")
    logging.info("    - fig3_margin_vs_accuracy.png ⭐ KEY FIGURE")
    logging.info("    - fig4_cdf_curves.png")
    logging.info("    - fig5_entropy_distribution.png")
    logging.info("    - fig6_gap_ratio_vs_f1.png")
    logging.info("  Log:")
    logging.info("    - analysis.log")
    
    logging.info("\n" + "="*80)


if __name__ == '__main__':
    main()