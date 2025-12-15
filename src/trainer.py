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

from utils import plot_losses, save_evaluation_results
from evaluation_v2 import Evaluator 

class ExperimentRunner:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, 
                 config: Dict, device: torch.device, run_output_dir: str, 
                 scheduler: Optional[_LRScheduler] = None):
        """
        初始化实验运行器
        
        Args:
            model: 神经网络模型
            optimizer: 优化器
            criterion: 损失函数
            config: 配置字典，包含 TRAIN_CONFIG, EVAL_CONFIG, TUNE_CONFIG, EXPERIMENT_MODE
            device: 计算设备
            run_output_dir: 输出目录
            scheduler: 学习率调度器（可选）
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.run_output_dir = run_output_dir
        self.scheduler = scheduler
        
        # 获取实验模式
        self.experiment_mode = config.get('EXPERIMENT_MODE', 'eval_hyperparam_search')
        
        self.scaler = GradScaler()
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf') 
        self.epochs_no_improve = 0
        self.current_epoch = 0
        
        self.best_model_path = os.path.join(self.run_output_dir, "best_model.pth")
        self.loss_plot_path = os.path.join(self.run_output_dir, "loss_curve.png")
        self.final_results_path = os.path.join(self.run_output_dir, "final_evaluation_results.json")

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        """
        内部方法：执行一轮训练
        
        Args:
            dataloader: 训练数据加载器
            
        Returns:
            float: 该轮的平均训练损失
        """
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
            
            # 梯度裁剪（如果配置了）
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
        """
        内部方法：计算验证集上的损失（非常快）
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            float: 验证集上的平均损失
        """
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
        """
        在验证集上搜索最佳的 K 和 Tau 超参数
        
        Args:
            evaluator: 评估器对象
            val_loader_metrics: 验证集数据加载器
            
        Returns:
            Tuple[int, float]: (最佳K值, 最佳Tau值)
        """
        logging.info("="*60)
        logging.info(" Starting Hyperparameter Tuning (k, tau) on Validation Set ")
        logging.info("="*60)
        
        # 定义搜索空间
        k_search_space = self.config['TUNE_CONFIG'].get('k_search', [3, 5, 7, 10, 15])
        tau_search_range = self.config['TUNE_CONFIG'].get('tau_search_range', [0.05, 0.60, 12])
        tau_search_space = np.linspace(*tau_search_range)
        
        logging.info(f"Search space:")
        logging.info(f"  k values: {k_search_space}")
        logging.info(f"  tau range: {tau_search_range[0]:.2f} to {tau_search_range[1]:.2f} ({tau_search_range[2]} points)")
        
        best_val_f1 = -1.0
        best_k = 0
        best_tau = 0.0
        
        tuning_pbar = tqdm(
            total=len(k_search_space) * len(tau_search_space), 
            desc="Tuning k/tau on Val Set"
        )

        for k in k_search_space:
            for tau in tau_search_space:
                # 在验证集上评估当前超参数组合
                # 注意：调优时不需要保存预测结果
                metrics = evaluator.evaluate(
                    val_loader_metrics, 
                    k=k, 
                    vote_tau=tau, 
                    query_set_name="Tuning", 
                    save_preds=False
                )
                
                # 使用 F1 分数作为选择标准
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

    def run(self, train_loader: DataLoader, val_loader_loss: DataLoader, 
            val_loader_metrics: DataLoader, gallery_loader: DataLoader, 
            ood_test_loaders: Dict[str, DataLoader]):
        """
        主训练循环（支持三种模式）
        
        Args:
            train_loader: 训练数据加载器
            val_loader_loss: 用于计算验证损失的数据加载器（可能为None）
            val_loader_metrics: 用于评估指标的验证数据加载器（可能为None）
            gallery_loader: 参考库数据加载器
            ood_test_loaders: OOD测试集数据加载器字典
        """
        epochs = self.config['TRAIN_CONFIG']['epochs']
        patience = self.config['TRAIN_CONFIG']['patience']
        
        # 检查是否有验证集
        has_validation = (val_loader_loss is not None and 
                         val_loader_metrics is not None and
                         len(val_loader_metrics.dataset) > 0)

        # ============================================================
        # Phase 1: 训练阶段
        # ============================================================
        logging.info("="*60)
        if self.experiment_mode == "train_hyperparam_search":
            logging.info(" PHASE 1: Training (Hyperparameter Search Mode) ")
        elif self.experiment_mode == "eval_hyperparam_search":
            logging.info(" PHASE 1: Training (with Train/Val Split) ")
        else:  # final_training
            logging.info(" PHASE 1: Training (on Full Dataset) ")
        logging.info("="*60)
        
        if has_validation:
            logging.info(f"Total Epochs: {epochs}, Early Stopping Patience: {patience}")
        else:
            logging.info(f"Total Epochs: {epochs} (No early stopping - using full data)")
        
        # 训练循环
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练一个 epoch
            train_loss = self._train_one_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if has_validation:
                # ========== 有验证集：计算验证损失并进行早停 ==========
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
                # ========== 无验证集：训练固定轮数 ==========
                current_lr = self.optimizer.param_groups[0]['lr']
                log_str = (f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"LR: {current_lr:.1e}")
                logging.info(log_str)
                
                # 学习率调度
                if self.scheduler:
                    self.scheduler.step()
                
                # 保存最后的模型（每个epoch都保存，或者只保存最后一个）
                if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                    torch.save(self.model.state_dict(), self.best_model_path)
                    logging.info(f"  Model checkpoint saved at epoch {epoch+1}")
        
        # 训练阶段结束
        logging.info("\n" + "="*60)
        logging.info(" PHASE 1: Training Finished ")
        logging.info("="*60)
        if has_validation:
            logging.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logging.info(f"Total epochs trained: {self.current_epoch + 1}")
        
        # 绘制损失曲线
        plot_losses(self.history, self.loss_plot_path)
        logging.info(f"Loss curve saved to: {self.loss_plot_path}")
        
        # 加载最佳模型
        if not os.path.exists(self.best_model_path):
            logging.error("No best model was saved. Cannot perform evaluation.")
            return

        logging.info(f"\nLoading best model from {self.best_model_path}...")
        self.model.load_state_dict(torch.load(self.best_model_path))
        
        # ============================================================
        # Phase 2 & 3: 根据实验模式决定后续操作
        # ============================================================
        
        if self.experiment_mode == "train_hyperparam_search":
            # ========== 模式1: 训练超参数搜索 ==========
            # 只需返回验证损失，不需要进行评估
            logging.info("\n" + "="*60)
            logging.info(" Mode: Train Hyperparameter Search ")
            logging.info(" Skipping evaluation phase ")
            logging.info("="*60)
            logging.info(f"Final result - Best validation loss: {self.best_val_loss:.4f}")
            return
        
        elif self.experiment_mode == "eval_hyperparam_search":
            # ========== 模式2: 评估超参数搜索 ==========
            # 需要在验证集上调优 k 和 tau
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
            
            # 保存找到的最佳评估超参数
            tuning_results = {
                'best_k': int(best_k),
                'best_tau': float(best_tau),
                'best_val_loss': float(self.best_val_loss)
            }
            tuning_save_path = os.path.join(self.run_output_dir, "best_eval_hyperparams.json")
            with open(tuning_save_path, 'w') as f:
                json.dump(tuning_results, f, indent=2)
            
            logging.info(f"\nBest evaluation hyperparameters saved to: {tuning_save_path}")
            
            # ========== Phase 3: 使用最佳超参数在验证集上做最终评估 ==========
            logging.info("\n" + "="*60)
            logging.info(f" PHASE 3: Final Evaluation on Validation Set ")
            logging.info(f" Using k={best_k}, tau={best_tau:.4f} ")
            logging.info("="*60)
            
            val_metrics = evaluator.evaluate(
                val_loader_metrics, 
                k=best_k, 
                vote_tau=best_tau,
                query_set_name="Validation Set", 
                save_preds=True, 
                output_dir=self.run_output_dir
            )
            
            logging.info("\nValidation Set Results:")
            for metric_name, score in val_metrics.items():
                logging.info(f"  {metric_name.capitalize()}: {score:.4f}")
            
            # 保存结果
            final_results = {"validation_set": val_metrics}
            save_evaluation_results(final_results, self.final_results_path)
            logging.info(f"\nResults saved to: {self.final_results_path}")
            
        else:  # final_training
            # ========== 模式3: 最终训练 ==========
            # 使用预设的评估超参数直接在测试集上评估
            logging.info("\n" + "="*60)
            logging.info(" PHASE 2: Skipped (Using Pre-determined Hyperparameters) ")
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
            
            # ========== Phase 3: 在所有测试集上进行最终评估 ==========
            logging.info("\n" + "="*60)
            logging.info(f" PHASE 3: Final Evaluation on Test Sets ")
            logging.info(f" Using k={best_k}, tau={best_tau:.4f} ")
            logging.info("="*60)
            
            final_results = {}
            
            # 评估所有 OOD 测试集
            for name, loader in ood_test_loaders.items():
                logging.info(f"\n{'─'*60}")
                logging.info(f" Evaluating on: {name.upper()} ")
                logging.info(f"{'─'*60}")
                
                test_metrics = evaluator.evaluate(
                    loader, 
                    k=best_k, 
                    vote_tau=best_tau,
                    query_set_name=name.upper(), 
                    save_preds=True, 
                    output_dir=self.run_output_dir
                )
                
                final_results[name] = test_metrics
                
                logging.info(f"\n{name.upper()} Results:")
                for metric_name, score in test_metrics.items():
                    logging.info(f"  {metric_name.capitalize()}: {score:.4f}")
            
            # 保存总结性的评估结果
            save_evaluation_results(final_results, self.final_results_path)
            logging.info(f"\nAll results saved to: {self.final_results_path}")
        
        # 实验完全结束
        logging.info("\n" + "="*60)
        logging.info(" EXPERIMENT COMPLETED SUCCESSFULLY ")
        logging.info("="*60)