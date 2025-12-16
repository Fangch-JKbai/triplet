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
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from utils import save_evaluation_results
from evaluation_v2 import Evaluator

class ExperimentRunnerWithMonitoring:
    """
    增强版训练器：每个epoch都在测试集上评估，实时绘制性能曲线
    """
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
                 config: Dict, device: torch.device, run_output_dir: str,
                 scheduler: Optional[_LRScheduler] = None,
                 eval_interval: int = 1):  # 每隔多少epoch评估一次
        """
        Args:
            eval_interval: 每隔多少epoch在测试集上评估一次（默认1，即每个epoch）
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.run_output_dir = run_output_dir
        self.scheduler = scheduler
        self.eval_interval = eval_interval

        # 获取实验模式
        self.experiment_mode = config.get('EXPERIMENT_MODE', 'final_training')

        self.scaler = GradScaler()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': [],
            # 测试集性能记录
            'price_auc': [],
            'price_f1': [],
            'new_auc': [],
            'new_f1': []
        }
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.current_epoch = 0

        self.best_model_path = os.path.join(self.run_output_dir, "best_model.pth")
        self.loss_plot_path = os.path.join(self.run_output_dir, "loss_curve.png")
        self.monitor_plot_path = os.path.join(self.run_output_dir, "test_performance_curve.png")
        self.history_json_path = os.path.join(self.run_output_dir, "training_history.json")
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

            # 梯度裁剪
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

    def _evaluate_test_sets(self, evaluator: Evaluator, ood_test_loaders: Dict[str, DataLoader],
                           k: int, vote_tau: float) -> Dict[str, Dict[str, float]]:
        """
        评估所有测试集

        Returns:
            Dict: {'price': {'auc': 0.7, 'f1': 0.5}, 'new': {...}}
        """
        results = {}

        for name, loader in ood_test_loaders.items():
            metrics = evaluator.evaluate(
                loader,
                k=k,
                vote_tau=vote_tau,
                query_set_name=name.upper(),
                save_preds=False  # 训练过程中不保存预测
            )
            results[name] = metrics

        return results

    def _plot_monitoring(self):
        """绘制训练过程中的测试集性能曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = self.history['epoch']

        # 1. AUC曲线
        ax1 = axes[0, 0]
        if self.history['price_auc']:
            ax1.plot(epochs, self.history['price_auc'], 'b-o', label='Price AUC', linewidth=2, markersize=4)
        if self.history['new_auc']:
            ax1.plot(epochs, self.history['new_auc'], 'r-s', label='New AUC', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('AUC', fontsize=12)
        ax1.set_title('Test Set AUC Over Epochs', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 2. F1曲线
        ax2 = axes[0, 1]
        if self.history['price_f1']:
            ax2.plot(epochs, self.history['price_f1'], 'b-o', label='Price F1', linewidth=2, markersize=4)
        if self.history['new_f1']:
            ax2.plot(epochs, self.history['new_f1'], 'r-s', label='New F1', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('Test Set F1 Over Epochs', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # 3. 训练/验证损失
        ax3 = axes[1, 0]
        if self.history['train_loss']:
            ax3.plot(self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            ax3.plot(self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        # 4. 综合对比（Price vs New的平均AUC）
        ax4 = axes[1, 1]
        if self.history['price_auc'] and self.history['new_auc']:
            avg_auc = [(p + n) / 2 for p, n in zip(self.history['price_auc'], self.history['new_auc'])]
            ax4.plot(epochs, avg_auc, 'g-o', label='Average AUC', linewidth=2, markersize=4)

            # 标注最佳epoch
            best_idx = np.argmax(avg_auc)
            best_epoch = epochs[best_idx]
            best_auc = avg_auc[best_idx]
            ax4.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
            ax4.scatter([best_epoch], [best_auc], color='red', s=100, zorder=5)
            ax4.text(best_epoch, best_auc, f'  {best_auc:.4f}', fontsize=10, verticalalignment='bottom')

        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Average AUC', fontsize=12)
        ax4.set_title('Average Test AUC (Best Epoch Marked)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.monitor_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"Monitoring plot saved to: {self.monitor_plot_path}")

    def _save_history(self):
        """保存训练历史到JSON"""
        with open(self.history_json_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logging.info(f"Training history saved to: {self.history_json_path}")

    def run(self, train_loader: DataLoader, val_loader_loss: DataLoader,
            val_loader_metrics: DataLoader, gallery_loader: DataLoader,
            ood_test_loaders: Dict[str, DataLoader]):
        """
        主训练循环（带测试集监控）
        """
        epochs = self.config['TRAIN_CONFIG']['epochs']
        patience = self.config['TRAIN_CONFIG']['patience']

        # 检查是否有验证集
        has_validation = (val_loader_loss is not None and
                         val_loader_metrics is not None and
                         len(val_loader_metrics.dataset) > 0)

        # 获取评估超参数
        default_k = self.config['EVAL_CONFIG'].get('default_k', 3)
        default_tau = self.config['EVAL_CONFIG'].get('default_tau', 0.35)

        # 创建评估器（只创建一次）
        logging.info("Creating evaluator for monitoring...")
        evaluator = Evaluator(
            self.model,
            gallery_loader,
            self.device,
            self.config['EVAL_CONFIG']
        )

        logging.info("="*60)
        logging.info(" TRAINING WITH TEST SET MONITORING ")
        logging.info("="*60)
        logging.info(f"Total Epochs: {epochs}")
        logging.info(f"Evaluation Interval: Every {self.eval_interval} epoch(s)")
        logging.info(f"Test sets: {list(ood_test_loaders.keys())}")
        logging.info(f"Using k={default_k}, tau={default_tau} for evaluation")
        if has_validation:
            logging.info(f"Early Stopping Patience: {patience}")
        logging.info("="*60)

        # 训练循环
        for epoch in range(epochs):
            self.current_epoch = epoch

            # 训练一个 epoch
            train_loss = self._train_one_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if has_validation:
                # 计算验证损失
                val_loss = self._evaluate_loss(val_loader_loss)
                self.history['val_loss'].append(val_loss)

                current_lr = self.optimizer.param_groups[0]['lr']
                log_str = (f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"LR: {current_lr:.1e}")
                logging.info(log_str)

                # 学习率调度
                if self.scheduler:
                    self.scheduler.step()

                # 早停逻辑
                if val_loss < self.best_val_loss:
                    improvement = self.best_val_loss - val_loss
                    logging.info(f"  ✓ New best validation loss: {val_loss:.4f} "
                               f"(improved by {improvement:.4f}). Saving model...")
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    logging.info(f"  ✗ No improvement for {self.epochs_no_improve} epoch(s)")

                if self.epochs_no_improve >= patience:
                    logging.info(f"\n{'='*60}")
                    logging.info(f"Early stopping triggered after {patience} epochs with no improvement")
                    logging.info(f"Best validation loss: {self.best_val_loss:.4f}")
                    logging.info(f"{'='*60}\n")
                    break

            else:
                # 无验证集：只记录训练损失
                current_lr = self.optimizer.param_groups[0]['lr']
                log_str = (f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"LR: {current_lr:.1e}")
                logging.info(log_str)

                if self.scheduler:
                    self.scheduler.step()

                # 定期保存模型
                if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                    torch.save(self.model.state_dict(), self.best_model_path)
                    logging.info(f"  Model checkpoint saved at epoch {epoch+1}")

            # ============================================================
            # 每隔eval_interval个epoch，在测试集上评估
            # ============================================================
            if (epoch + 1) % self.eval_interval == 0 or (epoch + 1) == epochs:
                logging.info(f"\n{'─'*60}")
                logging.info(f" Evaluating on test sets at epoch {epoch+1}")
                logging.info(f"{'─'*60}")

                test_results = self._evaluate_test_sets(
                    evaluator,
                    ood_test_loaders,
                    k=default_k,
                    vote_tau=default_tau
                )

                # 记录结果
                self.history['epoch'].append(epoch + 1)

                for name, metrics in test_results.items():
                    auc = metrics.get('auc', 0.0)
                    f1_key = f'f1_at_fixed_tau_{default_tau:.2f}'
                    f1 = metrics.get(f1_key, 0.0)

                    logging.info(f"  {name.upper()}: AUC={auc:.4f}, F1={f1:.4f}")

                    # 保存到历史记录
                    if name == 'price':
                        self.history['price_auc'].append(auc)
                        self.history['price_f1'].append(f1)
                    elif name == 'new':
                        self.history['new_auc'].append(auc)
                        self.history['new_f1'].append(f1)

                # 实时绘图
                self._plot_monitoring()

                # 保存历史记录
                self._save_history()

                logging.info(f"{'─'*60}\n")

        # 训练结束
        logging.info("\n" + "="*60)
        logging.info(" TRAINING FINISHED ")
        logging.info("="*60)
        if has_validation:
            logging.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logging.info(f"Total epochs trained: {self.current_epoch + 1}")

        # 最终绘图
        self._plot_monitoring()
        self._save_history()

        # 加载最佳模型
        if not os.path.exists(self.best_model_path):
            logging.error("No best model was saved. Cannot perform final evaluation.")
            return

        logging.info(f"\nLoading best model from {self.best_model_path}...")
        self.model.load_state_dict(torch.load(self.best_model_path))

        # 最终评估（保存预测）
        logging.info("\n" + "="*60)
        logging.info(" FINAL EVALUATION ON TEST SETS ")
        logging.info("="*60)

        final_results = {}
        for name, loader in ood_test_loaders.items():
            logging.info(f"\n{'─'*60}")
            logging.info(f" Evaluating on: {name.upper()} ")
            logging.info(f"{'─'*60}")

            test_metrics = evaluator.evaluate(
                loader,
                k=default_k,
                vote_tau=default_tau,
                query_set_name=name.upper(),
                save_preds=True,  # 最终评估保存预测
                output_dir=self.run_output_dir
            )

            final_results[name] = test_metrics

            logging.info(f"\n{name.upper()} Results:")
            for metric_name, score in test_metrics.items():
                logging.info(f"  {metric_name.capitalize()}: {score:.4f}")

        # 保存最终结果
        save_evaluation_results(final_results, self.final_results_path)
        logging.info(f"\nAll results saved to: {self.final_results_path}")

        # 找出最佳epoch
        if self.history['price_auc'] and self.history['new_auc']:
            avg_auc = [(p + n) / 2 for p, n in zip(self.history['price_auc'], self.history['new_auc'])]
            best_idx = np.argmax(avg_auc)
            best_epoch = self.history['epoch'][best_idx]
            best_avg_auc = avg_auc[best_idx]
            best_price_auc = self.history['price_auc'][best_idx]
            best_new_auc = self.history['new_auc'][best_idx]

            logging.info("\n" + "="*60)
            logging.info(" BEST EPOCH ANALYSIS ")
            logging.info("="*60)
            logging.info(f"Best Epoch (by average test AUC): {best_epoch}")
            logging.info(f"  Average AUC: {best_avg_auc:.4f}")
            logging.info(f"  Price AUC: {best_price_auc:.4f}")
            logging.info(f"  New AUC: {best_new_auc:.4f}")

        logging.info("\n" + "="*60)
        logging.info(" EXPERIMENT COMPLETED SUCCESSFULLY ")
        logging.info("="*60)
        logging.info(f"Check {self.monitor_plot_path} for performance curves")
