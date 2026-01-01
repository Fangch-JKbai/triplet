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
                 eval_interval: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.run_output_dir = run_output_dir
        self.scheduler = scheduler
        self.eval_interval = eval_interval

        self.scaler = GradScaler()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': [],
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
        self.model.eval()
        running_loss = 0.0
        for batch in dataloader:
            if batch is None: continue
            embeddings_1280 = batch['embeddings'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            with autocast(enabled=True):
                features_128 = self.model(embeddings_1280)
                loss = self.criterion(features_128, labels)
            running_loss += loss.item()
        return running_loss / max(1, len(dataloader))

    def _evaluate_test_sets(self, evaluator: Evaluator, ood_test_loaders: Dict[str, DataLoader],
                           k: int, vote_tau: float) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, loader in ood_test_loaders.items():
            metrics = evaluator.evaluate(
                loader,
                k=k,
                vote_tau=vote_tau,
                query_set_name=name.upper(),
                save_preds=False
            )
            results[name] = metrics
        return results

    def _plot_monitoring(self):
        # (保持原样，省略以节省空间，功能不变)
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            epochs = self.history['epoch']
            
            # 1. AUC
            ax1 = axes[0, 0]
            if self.history['price_auc']: ax1.plot(epochs, self.history['price_auc'], 'b-o', label='Price AUC')
            if self.history['new_auc']: ax1.plot(epochs, self.history['new_auc'], 'r-s', label='New AUC')
            ax1.set_title('Test Set AUC')
            ax1.legend()
            ax1.grid(True)

            # 2. F1
            ax2 = axes[0, 1]
            if self.history['price_f1']: ax2.plot(epochs, self.history['price_f1'], 'b-o', label='Price F1')
            if self.history['new_f1']: ax2.plot(epochs, self.history['new_f1'], 'r-s', label='New F1')
            ax2.set_title('Test Set F1')
            ax2.legend()
            ax2.grid(True)

            # 3. Loss
            ax3 = axes[1, 0]
            if self.history['train_loss']: ax3.plot(self.history['train_loss'], 'b-', label='Train Loss')
            if self.history['val_loss']: ax3.plot(self.history['val_loss'], 'r-', label='Val Loss')
            ax3.set_title('Loss')
            ax3.legend()
            ax3.grid(True)

            # 4. Avg
            ax4 = axes[1, 1]
            if self.history['price_auc'] and self.history['new_auc']:
                avg_auc = [(p + n) / 2 for p, n in zip(self.history['price_auc'], self.history['new_auc'])]
                ax4.plot(epochs, avg_auc, 'g-o', label='Avg AUC')
            ax4.set_title('Average AUC')
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()
            plt.savefig(self.monitor_plot_path)
            plt.close()
        except Exception as e:
            logging.warning(f"Plotting failed: {e}")

    def _save_history(self):
        with open(self.history_json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def run(self, train_loader: DataLoader, val_loader_loss: DataLoader,
            val_loader_metrics: DataLoader, gallery_loader: DataLoader,
            ood_test_loaders: Dict[str, DataLoader]):
        
        epochs = self.config['TRAIN_CONFIG']['epochs']
        patience = self.config['TRAIN_CONFIG']['patience']

        # 检查是否有验证集
        has_validation = (val_loader_loss is not None)

        default_k = self.config['EVAL_CONFIG'].get('default_k', 3)
        default_tau = self.config['EVAL_CONFIG'].get('default_tau', 0.35)

        # Bug 修复：移除此处的 Evaluator 初始化
        # evaluator = ... (已删除)

        logging.info("="*60)
        logging.info(" TRAINING WITH TEST SET MONITORING ")
        logging.info("="*60)
        logging.info(f"Total Epochs: {epochs}")
        if not has_validation:
            logging.info("Validation set not provided. Early stopping disabled.")
            logging.info("The model will be saved periodically and at the end.")
        
        for epoch in range(epochs):
            self.current_epoch = epoch

            # 1. 训练
            train_loss = self._train_one_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            current_lr = self.optimizer.param_groups[0]['lr']

            if has_validation:
                # 2a. 有验证集：执行早停逻辑
                val_loss = self._evaluate_loss(val_loader_loss)
                self.history['val_loss'].append(val_loss)

                log_str = (f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.1e}")
                logging.info(log_str)
                
                if self.scheduler: self.scheduler.step()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= patience:
                        logging.info("Early stopping triggered.")
                        break
            else:
                # 2b. 无验证集：保存最新模型，不早停
                log_str = (f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | LR: {current_lr:.1e}")
                logging.info(log_str)
                
                if self.scheduler: self.scheduler.step()
                
                # 更新 "最佳" 模型为当前模型，以便最后评估使用
                torch.save(self.model.state_dict(), self.best_model_path)
                
                # 定期 Checkpoint (可选)
                # if (epoch + 1) % 10 == 0:
                #     torch.save(self.model.state_dict(), os.path.join(self.run_output_dir, f"model_ep{epoch+1}.pth"))

            # ============================================================
            # 3. 评估逻辑 (BUG FIX)
            # ============================================================
            if (epoch + 1) % self.eval_interval == 0 or (epoch + 1) == epochs:
                logging.info(f"\nEvaluating on test sets at epoch {epoch+1}...")

                # --- FIX: 在这里初始化 Evaluator ---
                # 确保 Gallery 特征是基于当前模型参数重新计算的
                current_evaluator = Evaluator(
                    self.model,
                    gallery_loader,
                    self.device,
                    self.config['EVAL_CONFIG']
                )

                test_results = self._evaluate_test_sets(
                    current_evaluator, # 使用新的 evaluator
                    ood_test_loaders,
                    k=default_k,
                    vote_tau=default_tau
                )

                self.history['epoch'].append(epoch + 1)
                for name, metrics in test_results.items():
                    auc = metrics.get('auc', 0.0)
                    f1 = metrics.get(f'f1_at_fixed_tau_{default_tau:.2f}', 0.0)
                    logging.info(f"  {name.upper()}: AUC={auc:.4f}, F1={f1:.4f}")

                    if name == 'price':
                        self.history['price_auc'].append(auc)
                        self.history['price_f1'].append(f1)
                    elif name == 'new':
                        self.history['new_auc'].append(auc)
                        self.history['new_f1'].append(f1)

                self._plot_monitoring()
                self._save_history()

        # 训练结束
        logging.info("Training finished.")
        
        # 加载最佳模型（无验证集时即为最后保存的模型）
        if os.path.exists(self.best_model_path):
            logging.info(f"Loading best model from {self.best_model_path} for final evaluation...")
            self.model.load_state_dict(torch.load(self.best_model_path))
            
            # 创建最终的 Evaluator
            final_evaluator = Evaluator(
                self.model,
                gallery_loader,
                self.device,
                self.config['EVAL_CONFIG']
            )

            logging.info("FINAL EVALUATION ON TEST SETS")
            final_results = {}
            for name, loader in ood_test_loaders.items():
                test_metrics = final_evaluator.evaluate(
                    loader,
                    k=default_k,
                    vote_tau=default_tau,
                    query_set_name=name.upper(),
                    save_preds=True,
                    output_dir=self.run_output_dir
                )
                final_results[name] = test_metrics
            
            save_evaluation_results(final_results, self.final_results_path)