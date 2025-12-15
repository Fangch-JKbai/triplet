# trainer_v3.py (综合版本)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
from typing import Dict, List, Tuple
import json

from utils import plot_losses, save_evaluation_results
from evaluation_v3 import ComprehensiveEvaluator
from intrinsic_evaluation import IntrinsicEmbeddingEvaluator
import visualization as viz

class ComprehensiveExperimentRunner:
    """
    综合实验运行器：
    1. 训练阶段
    2. 超参数调优
    3. 下游任务评估
    4. Embedding内在质量评估
    5. 可视化生成
    """
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 criterion: nn.Module, config: Dict, device: torch.device, 
                 run_output_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.run_output_dir = run_output_dir
        
        self.scaler = GradScaler()
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.current_epoch = 0
        
        # 路径配置
        self.best_model_path = os.path.join(run_output_dir, "best_model.pth")
        self.loss_plot_path = os.path.join(run_output_dir, "loss_curve.png")
        self.final_results_path = os.path.join(run_output_dir, "final_evaluation_results.json")
        
        # 可视化目录
        self.viz_dir = os.path.join(run_output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} Training")
        for batch in progress_bar:
            if batch is None:
                continue
            embeddings_1280 = batch['embeddings'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                features_128 = self.model(embeddings_1280)
                loss = self.criterion(features_128, labels)
            
            self.scaler.scale(loss).backward()
            grad_clip_norm = self.config['TRAIN_CONFIG'].get("grad_clip_norm", 0)
            if grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            item_loss = loss.item()
            running_loss += item_loss
            progress_bar.set_postfix(loss=f"{item_loss:.4f}")
        
        return running_loss / max(1, len(dataloader))
    
    @torch.no_grad()
    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        """计算验证集损失"""
        self.model.eval()
        running_loss = 0.0
        for batch in dataloader:
            if batch is None:
                continue
            embeddings_1280 = batch['embeddings'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            with autocast(enabled=True):
                features_128 = self.model(embeddings_1280)
                loss = self.criterion(features_128, labels)
            running_loss += loss.item()
        return running_loss / max(1, len(dataloader))
    
    def run(self, train_loader: DataLoader, val_loader_loss: DataLoader, 
            val_loader_metrics: DataLoader, gallery_loader: DataLoader, 
            ood_test_loaders: Dict[str, DataLoader]):
        """主实验流程"""
        
        # =============================================================
        # Phase 1: 训练
        # =============================================================
        logging.info("="*60)
        logging.info("PHASE 1: Model Training")
        logging.info("="*60)
        
        epochs = self.config['TRAIN_CONFIG']['epochs']
        patience = self.config['TRAIN_CONFIG']['patience']
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            train_loss = self._train_one_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            val_loss = self._evaluate_loss(val_loader_loss)
            self.history['val_loss'].append(val_loss)
            
            logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                logging.info(f"  → New best val loss: {val_loss:.4f} (prev: {self.best_val_loss:.4f})")
                torch.save(self.model.state_dict(), self.best_model_path)
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            if self.epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after {patience} epochs.")
                break
        
        plot_losses(self.history, self.loss_plot_path)
        
        # 加载最佳模型
        logging.info(f"Loading best model from {self.best_model_path}")
        self.model.load_state_dict(torch.load(self.best_model_path))
        
        # =============================================================
        # Phase 2: 超参数调优
        # =============================================================
        logging.info("="*60)
        logging.info("PHASE 2: Hyperparameter Tuning")
        logging.info("="*60)
        
        evaluator = ComprehensiveEvaluator(
            self.model, gallery_loader, self.device, 
            self.config['EVAL_CONFIG']
        )
        
        k_search = self.config['TUNE_CONFIG']['k_search']
        tau_range = self.config['TUNE_CONFIG']['tau_search_range']
        tau_search = np.linspace(*tau_range)
        
        best_k, best_tau, tuning_results = evaluator.tune_hyperparameters(
            val_loader_metrics, k_search, tau_search
        )
        
        # 保存调优结果
        tuning_save_path = os.path.join(self.run_output_dir, "tuning_results.json")
        with open(tuning_save_path, 'w') as f:
            json.dump(tuning_results, f, indent=4)
        
        # 生成调优热图
        if self.config.get('VISUALIZATION_CONFIG', {}).get('performance_heatmap', {}).get('enable', False):
            heatmap_path = os.path.join(self.viz_dir, "tuning_heatmap.png")
            viz.plot_tuning_heatmap(
                tuning_results['k_values'],
                tuning_results['tau_values'],
                np.array(tuning_results['heatmap']),
                heatmap_path,
                best_k, best_tau
            )
        
        # =============================================================
        # Phase 3: 下游任务评估
        # =============================================================
        logging.info("="*60)
        logging.info(f"PHASE 3: Downstream Task Evaluation (k={best_k}, tau={best_tau:.4f})")
        logging.info("="*60)
        
        final_results = {}
        
        # 验证集
        logging.info("Evaluating on Validation Set...")
        val_metrics = evaluator.evaluate_downstream_task(
            val_loader_metrics, k=best_k, vote_tau=best_tau,
            query_set_name="Validation Set", save_preds=True,
            output_dir=self.run_output_dir
        )
        final_results["validation_set"] = val_metrics
        self._log_metrics(val_metrics, "Validation Set")
        
        # OOD测试集
        for name, loader in ood_test_loaders.items():
            logging.info(f"Evaluating on OOD Test Set: {name.upper()}...")
            test_metrics = evaluator.evaluate_downstream_task(
                loader, k=best_k, vote_tau=best_tau,
                query_set_name=name.upper(), save_preds=True,
                output_dir=self.run_output_dir
            )
            final_results[name] = test_metrics
            self._log_metrics(test_metrics, name.upper())
        
        # =============================================================
        # Phase 4: Embedding内在质量评估
        # =============================================================
        intrinsic_config = self.config.get('INTRINSIC_EVAL_CONFIG', {})
        if intrinsic_config.get('enable', False):
            logging.info("="*60)
            logging.info("PHASE 4: Intrinsic Embedding Quality Evaluation")
            logging.info("="*60)
            
            intrinsic_evaluator = IntrinsicEmbeddingEvaluator(intrinsic_config)
            
            # 对每个测试集进行内在评估
            for set_name, loader in [("validation", val_loader_metrics)] + list(ood_test_loaders.items()):
                logging.info(f"\n--- Intrinsic Evaluation: {set_name.upper()} ---")
                
                # 获取embeddings
                query_embs, query_labels, query_ids = evaluator._get_all_embeddings_and_labels(
                    loader, f"Extracting {set_name}"
                )
                
                intrinsic_results = intrinsic_evaluator.full_intrinsic_evaluation(
                    query_embs, query_labels, query_ids,
                    evaluator.gallery_embeddings, 
                    evaluator.gallery_labels,
                    evaluator.gallery_ids
                )
                
                # 合并到最终结果中
                final_results[f"{set_name}_intrinsic"] = intrinsic_results
                
                # 生成可视化
                self._generate_intrinsic_visualizations(
                    query_embs, query_labels, 
                    intrinsic_results, set_name
                )
        
        # =============================================================
        # Phase 5: 生成综合报告
        # =============================================================
        logging.info("="*60)
        logging.info("PHASE 5: Generating Comprehensive Report")
        logging.info("="*60)
        
        # 保存所有结果
        save_evaluation_results(final_results, self.final_results_path)
        
        # 生成综合报告图
        if self.config.get('VISUALIZATION_CONFIG', {}).get('enable', False):
            # 合并验证集的下游任务和内在指标
            combined_metrics = {**final_results.get('validation_set', {}), 
                              **final_results.get('validation_intrinsic', {})}
            
            report_fig_path = os.path.join(self.viz_dir, "comprehensive_report.png")
            viz.create_comprehensive_report_figure(combined_metrics, report_fig_path)
        
        logging.info("="*60)
        logging.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logging.info("="*60)
        logging.info(f"Results saved to: {self.final_results_path}")
        logging.info(f"Visualizations saved to: {self.viz_dir}")
    
    def _log_metrics(self, metrics: Dict, set_name: str):
        """辅助函数：打印指标"""
        for metric_name, score in metrics.items():
            if isinstance(score, (int, float)):
                logging.info(f"  {metric_name}: {score:.4f}")
    
    def _generate_intrinsic_visualizations(self, embeddings: torch.Tensor,
                                          labels: List[List[str]],
                                          results: Dict, set_name: str):
        """生成内在评估相关的可视化"""
        viz_config = self.config.get('VISUALIZATION_CONFIG', {})
        
        # 转为numpy
        embs_np = embeddings.cpu().numpy()
        # 取第一个标签作为可视化用途
        single_labels = [label_list[0] if label_list else 'unknown' 
                        for label_list in labels]
        
        # t-SNE
        if viz_config.get('tsne', {}).get('enable', False):
            tsne_path = os.path.join(self.viz_dir, f"tsne_{set_name}.png")
            viz.plot_tsne(embs_np, single_labels, tsne_path,
                         max_samples=viz_config['tsne'].get('max_samples', 2000),
                         title=f"t-SNE: {set_name.upper()}")
        
        # 相似度分布
        if 'positive_sims_sample' in results:
            sim_dist_path = os.path.join(self.viz_dir, 
                                        f"similarity_distribution_{set_name}.png")
            viz.plot_similarity_distribution(
                results['positive_sims_sample'],
                results['negative_sims_sample'],
                sim_dist_path
            )
        
        # EC频率性能
        if 'common_accuracy' in results:
            freq_perf_path = os.path.join(self.viz_dir, 
                                         f"ec_frequency_performance_{set_name}.png")
            viz.plot_ec_frequency_performance(results, freq_perf_path)
        
        # k-NN准确率曲线
        if any(k.startswith('knn_accuracy_k') for k in results.keys()):
            knn_curve_path = os.path.join(self.viz_dir, 
                                         f"knn_accuracy_curve_{set_name}.png")
            viz.plot_knn_accuracy_curve(results, knn_curve_path)