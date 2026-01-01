# trainer.py (完整版 - 支持训练损失早停)
"""
实验运行器 - 优化版
- 集成AUPRC曲线绘制
- 消除评估时的重复推理 (Double Inference Removal)
- 支持多测试集PR曲线对比图
- 支持基于训练损失的早停（无验证集场景）
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler 
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from utils import plot_losses, save_evaluation_results
from evaluation import Evaluator
from collections import defaultdict


class ExperimentRunner:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, 
                 config: Dict, device: torch.device, run_output_dir: str, 
                 scheduler: Optional[_LRScheduler] = None):
        """
        初始化实验运行器
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.run_output_dir = run_output_dir
        self.scheduler = scheduler
        
        self.experiment_mode = config.get('EXPERIMENT_MODE', 'hyperparameter_search')
        
        self.scaler = GradScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')  # 新增：记录最佳训练损失
        self.epochs_no_improve = 0
        self.current_epoch = 0
        
        self.best_model_path = os.path.join(self.run_output_dir, "best_model.pth")
        self.loss_plot_path = os.path.join(self.run_output_dir, "loss_curve.png")
        self.lr_plot_path = os.path.join(self.run_output_dir, "lr_schedule.png")
        self.final_results_path = os.path.join(self.run_output_dir, "final_evaluation_results.json")

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        """执行一轮训练"""
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
        """计算验证集上的损失"""
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

    def _tune_k_and_tau(self, evaluator: Evaluator, val_loader_metrics: DataLoader) -> Tuple[int, float]:
        """在验证集上搜索最佳的 K 和 Tau 超参数"""
        logging.info("="*60)
        logging.info(" Starting Hyperparameter Tuning (k, tau) on Validation Set ")
        logging.info("="*60)
        
        k_search_space = self.config['TUNE_CONFIG'].get('k_search', [3, 5, 7, 10, 15])
        tau_search_range = self.config['TUNE_CONFIG'].get('tau_search_range', [0.05, 0.60, 12])
        tau_search_space = np.linspace(*tau_search_range)
        
        logging.info(f"Search space:")
        logging.info(f"  k values: {k_search_space}")
        logging.info(f"  tau range: {tau_search_range[0]:.2f} to {tau_search_range[1]:.2f} ({int(tau_search_range[2])} points)")
        
        best_val_f1 = -1.0
        best_k = 0
        best_tau = 0.0
        
        tuning_pbar = tqdm(
            total=len(k_search_space) * len(tau_search_space), 
            desc="Tuning k/tau on Val Set"
        )

        for k in k_search_space:
            for tau in tau_search_space:
                # 调优阶段不需要 raw scores，只关注 F1
                metrics = evaluator.evaluate(
                    val_loader_metrics, 
                    k=k, 
                    vote_tau=tau, 
                    query_set_name="Tuning", 
                    save_preds=False,
                    plot_auprc=False,
                    return_raw_scores=False 
                )
                
                current_f1 = metrics.get('f1', 0.0)
                
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_k = k
                    best_tau = tau
                
                tuning_pbar.set_postfix(
                    k=k, 
                    tau=f"{tau:.2f}", 
                    f1=f"{current_f1:.4f}",
                    best_f1=f"{best_val_f1:.4f}"
                )
                tuning_pbar.update(1)
        
        tuning_pbar.close()
        
        logging.info("="*60)
        logging.info(f" Tuning Finished ")
        logging.info(f"  Best k: {best_k}")
        logging.info(f"  Best tau: {best_tau:.4f}")
        logging.info(f"  Best Val F1: {best_val_f1:.4f}")
        logging.info("="*60)
        
        return best_k, best_tau

    def _plot_lr_history(self):
        """绘制学习率变化曲线"""
        if not self.history['learning_rate']:
            return
            
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(self.history['learning_rate']) + 1), 
                self.history['learning_rate'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.lr_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Learning rate curve saved to: {self.lr_plot_path}")

    def _evaluate_and_collect_pr_data(
        self, 
        evaluator: Evaluator, 
        loader: DataLoader, 
        k: int, 
        tau: float,
        set_name: str
    ) -> Tuple[Dict[str, float], Optional[Tuple[np.ndarray, np.ndarray, float]]]:
        """
        评估并收集PR曲线数据 (优化版：单次推理)
        
        Returns:
            (metrics_dict, (y_true, y_score, auprc) or None)
        """
        # === 核心优化：只调用一次 evaluate，请求 raw_scores ===
        eval_output = evaluator.evaluate(
            loader, 
            k=k, 
            vote_tau=tau,
            query_set_name=set_name, 
            save_preds=True, 
            output_dir=self.run_output_dir,
            plot_auprc=False, # 此标志在 evaluator 内部已废弃，由 return_raw_scores 控制
            return_raw_scores=True
        )
        
        # 1. 提取 Metrics (排除 raw_data)
        metrics = {k: v for k, v in eval_output.items() if k != 'raw_data'}
        
        # 2. 提取 PR Data 并计算 AUPRC
        pr_data = None
        if 'raw_data' in eval_output:
            y_true, y_score = eval_output['raw_data']
            
            # 只有当存在有效数据时才处理
            if len(y_true) > 0:
                try:
                    auprc = average_precision_score(y_true, y_score)
                    pr_data = (y_true, y_score, auprc)
                except Exception as e:
                    logging.warning(f"Failed to calculate AUPRC for {set_name}: {e}")
        
        return metrics, pr_data

    def _plot_combined_pr_curves(
        self, 
        pr_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]]
    ):
        """绘制多测试集的PR曲线对比图"""
        if not pr_data_dict:
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 使用不同颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, (set_name, (y_true, y_score, auprc)) in enumerate(pr_data_dict.items()):
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            color = colors[idx % len(colors)]
            
            ax.plot(recall, precision, color=color, linewidth=2.5,
                    label=f'{set_name} (AUPRC = {auprc:.4f})')
            
            # 绘制baseline
            baseline = np.sum(y_true) / len(y_true)
            ax.axhline(y=baseline, color=color, linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=16)
        ax.legend(loc='upper right', fontsize=11)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.run_output_dir, "auprc_curves_comparison.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Combined PR curves saved to: {save_path}")

    def run(self, train_loader: DataLoader, val_loader_loss: DataLoader, 
            val_loader_metrics: DataLoader, gallery_loader: DataLoader, 
            ood_test_loaders: Dict[str, DataLoader]):
        """主训练循环"""
        epochs = self.config['TRAIN_CONFIG']['epochs']
        patience = self.config['TRAIN_CONFIG']['patience']
        
        has_validation = (val_loader_loss is not None and 
                         val_loader_metrics is not None and
                         len(val_loader_metrics.dataset) > 0)

        # ============================================================
        # Phase 1: 训练阶段
        # ============================================================
        logging.info("="*60)
        if self.experiment_mode == "hyperparameter_search":
            logging.info(" PHASE 1: Training (with Train/Val Split) ")
        else:
            logging.info(" PHASE 1: Training (on Full Dataset) ")
        logging.info("="*60)
        
        # 计算早停阈值（80%）
        early_stop_threshold = int(epochs * 0.8)
        
        if has_validation:
            logging.info(f"Total Epochs: {epochs}, Early Stopping Patience: {patience}")
            logging.info(f"Early stopping (based on validation loss) will be enabled after epoch {early_stop_threshold + 1} (80% of total)")
        else:
            logging.info(f"Total Epochs: {epochs}, Early Stopping Patience: {patience}")
            logging.info(f"Early stopping (based on training loss) will be enabled after epoch {early_stop_threshold + 1} (80% of total)")
        
        # 训练循环
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # 训练一个 epoch
            train_loss = self._train_one_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if has_validation:
                # ========== 有验证集：基于验证损失早停 ==========
                val_loss = self._evaluate_loss(val_loader_loss)
                self.history['val_loss'].append(val_loss)
                
                log_str = (f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"LR: {current_lr:.2e}")
                logging.info(log_str)
                
                # 学习率调度（在epoch结束后）
                if self.scheduler:
                    self.scheduler.step()

                # === 核心修改：80%之后才启用最佳模型更新和早停逻辑 ===
                if epoch >= early_stop_threshold:
                    # 从80%开始，正常执行早停逻辑
                    if val_loss < self.best_val_loss:
                        improvement = self.best_val_loss - val_loss if self.best_val_loss != float('inf') else val_loss
                        
                        # 首次更新时特别标注
                        if self.best_val_loss == float('inf'):
                            logging.info(f"  ✓ First best model recorded at epoch {epoch+1}: {val_loss:.4f}. Saving model...")
                        else:
                            logging.info(f"  ✓ New best validation loss: {val_loss:.4f} "
                                       f"(improved by {improvement:.4f}). Saving model...")
                        
                        torch.save(self.model.state_dict(), self.best_model_path)
                        self.best_val_loss = val_loss
                        self.epochs_no_improve = 0
                    else:
                        # 只有当best_val_loss不是默认值时，才累积耐心值
                        if self.best_val_loss != float('inf'):
                            self.epochs_no_improve += 1
                            logging.info(f"  ✗ No improvement for {self.epochs_no_improve} epoch(s)")
                        else:
                            # 还没有记录过最佳模型，不累积耐心值
                            logging.info(f"  ⊙ Val loss: {val_loss:.4f} (waiting for first best model)")
                    
                    # 只有在已记录最佳模型后才检查早停
                    if self.best_val_loss != float('inf') and self.epochs_no_improve >= patience:
                        logging.info(f"\n{'='*60}")
                        logging.info(f"Early stopping triggered after {patience} epochs with no improvement")
                        logging.info(f"(Early stopping enabled after epoch {early_stop_threshold + 1})")
                        logging.info(f"Best validation loss: {self.best_val_loss:.4f}")
                        logging.info(f"{'='*60}\n")
                        break
                else:
                    # 前80%：只记录信息，不更新best_val_loss和耐心值
                    logging.info(f"  ⊙ Val loss: {val_loss:.4f} (early stopping disabled until epoch {early_stop_threshold + 1})")
                    
            else:
                # ========== 无验证集：基于训练损失早停 ==========
                log_str = (f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"LR: {current_lr:.2e}")
                logging.info(log_str)
                
                if self.scheduler:
                    self.scheduler.step()
                
                # === 新增：基于训练损失的早停逻辑 ===
                if epoch >= early_stop_threshold:
                    # 从80%开始监控训练损失
                    if train_loss < self.best_train_loss:
                        improvement = self.best_train_loss - train_loss if self.best_train_loss != float('inf') else train_loss
                        
                        # 首次更新时特别标注
                        if self.best_train_loss == float('inf'):
                            logging.info(f"  ✓ First best model recorded at epoch {epoch+1}: {train_loss:.4f}. Saving model...")
                        else:
                            logging.info(f"  ✓ New best training loss: {train_loss:.4f} "
                                       f"(improved by {improvement:.4f}). Saving model...")
                        
                        torch.save(self.model.state_dict(), self.best_model_path)
                        self.best_train_loss = train_loss
                        self.epochs_no_improve = 0
                    else:
                        # 只有当best_train_loss不是默认值时，才累积耐心值
                        if self.best_train_loss != float('inf'):
                            self.epochs_no_improve += 1
                            logging.info(f"  ✗ No improvement for {self.epochs_no_improve} epoch(s)")
                        else:
                            # 还没有记录过最佳模型，不累积耐心值
                            logging.info(f"  ⊙ Train loss: {train_loss:.4f} (waiting for first best model)")
                    
                    # 只有在已记录最佳模型后才检查早停
                    if self.best_train_loss != float('inf') and self.epochs_no_improve >= patience:
                        logging.info(f"\n{'='*60}")
                        logging.info(f"Early stopping triggered after {patience} epochs with no improvement")
                        logging.info(f"(Early stopping enabled after epoch {early_stop_threshold + 1})")
                        logging.info(f"Best training loss: {self.best_train_loss:.4f}")
                        logging.info(f"{'='*60}\n")
                        break
                else:
                    # 前80%：只记录信息，不更新best_train_loss和耐心值
                    logging.info(f"  ⊙ Train loss: {train_loss:.4f} (early stopping disabled until epoch {early_stop_threshold + 1})")
        
        # 训练阶段结束
        logging.info("\n" + "="*60)
        logging.info(" PHASE 1: Training Finished ")
        logging.info("="*60)
        if has_validation:
            if self.best_val_loss != float('inf'):
                logging.info(f"Best validation loss: {self.best_val_loss:.4f}")
            else:
                logging.info("Note: Training ended before recording any best model (80% threshold not reached)")
        else:
            if self.best_train_loss != float('inf'):
                logging.info(f"Best training loss: {self.best_train_loss:.4f}")
            else:
                logging.info("Note: Training ended before recording any best model (80% threshold not reached)")
        logging.info(f"Total epochs trained: {self.current_epoch + 1}")
        
        # 绘制曲线
        plot_losses(self.history, self.loss_plot_path)
        logging.info(f"Loss curve saved to: {self.loss_plot_path}")
        self._plot_lr_history()
        
        # 加载最佳模型
        if not os.path.exists(self.best_model_path):
            logging.error("No best model was saved. Cannot perform evaluation.")
            return

        logging.info(f"\nLoading best model from {self.best_model_path}...")
        self.model.load_state_dict(torch.load(self.best_model_path))
        
        # ============================================================
        # Phase 2 & 3: 评估阶段
        # ============================================================
        
        if self.experiment_mode == "hyperparameter_search":
            # ========== 超参数搜索模式 ==========
            logging.info("\n" + "="*60)
            logging.info(" PHASE 2: Evaluation Hyperparameter Tuning ")
            logging.info("="*60)
            
            evaluator = Evaluator(
                self.model, 
                gallery_loader, 
                self.device, 
                self.config['EVAL_CONFIG']
            )
            
            best_k, best_tau = self._tune_k_and_tau(evaluator, val_loader_metrics)
            
            tuning_results = {
                'best_k': int(best_k),
                'best_tau': float(best_tau),
                'best_val_loss': float(self.best_val_loss)
            }
            tuning_save_path = os.path.join(self.run_output_dir, "best_eval_hyperparams.json")
            with open(tuning_save_path, 'w') as f:
                json.dump(tuning_results, f, indent=2)
            
            logging.info(f"\nBest evaluation hyperparameters saved to: {tuning_save_path}")
            
            # Phase 3: 验证集最终评估
            logging.info("\n" + "="*60)
            logging.info(f" PHASE 3: Final Evaluation on Validation Set ")
            logging.info(f" Using k={best_k}, tau={best_tau:.4f} ")
            logging.info("="*60)
            
            val_metrics = evaluator.evaluate(
                val_loader_metrics, 
                k=best_k, 
                vote_tau=best_tau,
                query_set_name="Validation_Set", 
                save_preds=True, 
                output_dir=self.run_output_dir,
                plot_auprc=True,
                return_raw_scores=True # 验证集也画一下PR曲线
            )
            
            logging.info("\nValidation Set Results:")
            for metric_name, score in val_metrics.items():
                if metric_name != 'raw_data':
                    logging.info(f"  {metric_name.upper()}: {score:.4f}")
            
            # 保存 metrics (不含 raw_data)
            final_metrics_to_save = {k: v for k, v in val_metrics.items() if k != 'raw_data'}
            save_evaluation_results({"validation_set": final_metrics_to_save}, self.final_results_path)
            logging.info(f"\nResults saved to: {self.final_results_path}")
            
        else:  # final_training
            # ========== 最终训练模式 ==========
            logging.info("\n" + "="*60)
            logging.info(" PHASE 2: Using Pre-determined Hyperparameters ")
            logging.info("="*60)
            
            best_k = self.config['EVAL_CONFIG'].get('default_k', 3)
            best_tau = self.config['EVAL_CONFIG'].get('default_tau', 0.35)
            
            logging.info(f"Using pre-determined hyperparameters:")
            logging.info(f"  k = {best_k}")
            logging.info(f"  tau = {best_tau:.4f}")
            
            evaluator = Evaluator(
                self.model, 
                gallery_loader, 
                self.device, 
                self.config['EVAL_CONFIG']
            )
            
            # Phase 3: 在所有测试集上进行最终评估，并收集PR数据
            logging.info("\n" + "="*60)
            logging.info(f" PHASE 3: Final Evaluation on Test Sets ")
            logging.info(f" Using k={best_k}, tau={best_tau:.4f} ")
            logging.info("="*60)
            
            final_results = {}
            pr_data_collection = {}  # 收集所有测试集的PR数据
            
            for name, loader in ood_test_loaders.items():
                logging.info(f"\n{'─'*60}")
                logging.info(f" Evaluating on: {name.upper()} ")
                logging.info(f"{'─'*60}")
                
                # 调用优化后的单次推理方法
                test_metrics, pr_data = self._evaluate_and_collect_pr_data(
                    evaluator, loader, best_k, best_tau, name.upper()
                )
                
                final_results[name] = test_metrics
                if pr_data is not None:
                    pr_data_collection[name.upper()] = pr_data
                
                logging.info(f"\n{name.upper()} Results:")
                for metric_name, score in test_metrics.items():
                    logging.info(f"  {metric_name.upper()}: {score:.4f}")
            
            # 绘制多测试集PR曲线对比图
            if pr_data_collection:
                self._plot_combined_pr_curves(pr_data_collection)
            
            # 保存结果
            save_evaluation_results(final_results, self.final_results_path)
            logging.info(f"\nAll results saved to: {self.final_results_path}")
        
        logging.info("\n" + "="*60)
        logging.info(" EXPERIMENT COMPLETED SUCCESSFULLY ")
        logging.info("="*60)