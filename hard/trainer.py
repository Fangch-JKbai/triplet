# triplet/hard/trainer.py
"""
训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import os
import logging
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 兼容不同 PyTorch 版本的 AMP API
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = 'cuda'
    USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = None  # 旧版本不需要指定
    USE_NEW_AMP = False


class Trainer:
    """难负样本学习训练器"""
    
    def __init__(self, model, criterion, optimizer, scheduler, device, config, output_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.output_dir = output_dir
        
        # 兼容不同 PyTorch 版本
        if USE_NEW_AMP:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = GradScaler()
        
        # 训练历史
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_f1_epochs": [],  # 记录评估发生的 epoch
            "learning_rate": []
        }
        
        self.best_val_f1 = -1
        self.best_epoch = 0
        self.epochs_no_improve = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _autocast_context(self):
        """返回适合当前 PyTorch 版本的 autocast context"""
        if USE_NEW_AMP:
            return autocast(device_type='cuda', enabled=True)
        else:
            return autocast(enabled=True)
    
    def train_epoch(self, dataloader, sampler, epoch):
        """训练一个epoch"""
        self.model.train()
        sampler.set_epoch(epoch)  # 更新采样策略
        
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
        for batch in pbar:
            embeddings_1280 = batch["embeddings"].to(self.device)
            primary_ecs = batch["primary_ecs"]
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with self._autocast_context():
                embeddings_128 = self.model(embeddings_1280)
                loss = self.criterion(embeddings_128, primary_ecs)
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config["TRAIN_CONFIG"]["grad_clip_norm"] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["TRAIN_CONFIG"]["grad_clip_norm"]
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = running_loss / max(num_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def evaluate_loss(self, dataloader):
        """评估验证集损失"""
        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            embeddings_1280 = batch["embeddings"].to(self.device)
            primary_ecs = batch["primary_ecs"]
            
            with self._autocast_context():
                embeddings_128 = self.model(embeddings_1280)
                loss = self.criterion(embeddings_128, primary_ecs)
            
            running_loss += loss.item()
            num_batches += 1
        
        avg_loss = running_loss / max(num_batches, 1)
        return avg_loss
    
    def record_val_f1(self, epoch, f1):
        """记录验证 F1，同时记录对应的 epoch"""
        self.history["val_f1"].append(f1)
        self.history["val_f1_epochs"].append(epoch)
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "history": self.history
        }
        
        if is_best:
            path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, path)
            logging.info(f"  ✓ Saved best model (F1={self.best_val_f1:.4f})")
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded checkpoint from {path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history["train_loss"], label="Train Loss", color='blue')
        if self.history["val_loss"]:
            axes[0, 0].plot(epochs, self.history["val_loss"], label="Val Loss", color='orange')
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 - 使用记录的 epoch 位置
        if self.history["val_f1"] and self.history["val_f1_epochs"]:
            # 转换为 1-indexed 用于显示
            f1_epochs = [e + 1 for e in self.history["val_f1_epochs"]]
            axes[0, 1].plot(f1_epochs, self.history["val_f1"], 'go-', 
                          label="Val F1", linewidth=2, markersize=6)
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("F1 Score")
            axes[0, 1].set_title("Validation F1 Score")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 设置 x 轴范围与 loss 图一致
            axes[0, 1].set_xlim([0, len(self.history["train_loss"]) + 1])
        
        # Learning Rate
        if self.history["learning_rate"]:
            axes[1, 0].plot(epochs, self.history["learning_rate"], label="Learning Rate", color='purple')
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("LR")
            axes[1, 0].set_title("Learning Rate Schedule")
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 第四个子图：Loss vs F1 对比（双 y 轴）
        if self.history["val_f1"] and self.history["val_f1_epochs"]:
            ax1 = axes[1, 1]
            ax2 = ax1.twinx()
            
            # 绘制 train loss
            line1 = ax1.plot(epochs, self.history["train_loss"], 'b-', label="Train Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # 绘制 F1
            f1_epochs = [e + 1 for e in self.history["val_f1_epochs"]]
            line2 = ax2.plot(f1_epochs, self.history["val_f1"], 'g-o', label="Val F1")
            ax2.set_ylabel("F1 Score", color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='center right')
            ax1.set_title("Loss vs F1 Score")
            ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        
        logging.info(f"Training curves saved to {save_path}")
    
    def save_results(self, results):
        """保存评估结果"""
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        
        def convert_to_serializable(obj):
            """递归转换 numpy 类型为 Python 原生类型"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, 'item'):  # torch tensor
                return obj.item()
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logging.info(f"Results saved to {results_path}")


def create_scheduler(optimizer, train_config):
    """
    创建学习率调度器
    
    Args:
        optimizer: PyTorch optimizer
        train_config: dict, 训练配置 (直接传递 TRAIN_CONFIG)
    """
    scheduler_type = train_config.get("scheduler_type", "cosine_warmup")
    total_epochs = train_config.get("epochs", 500)
    warmup_ratio = train_config.get("warmup_ratio", 0.1)
    min_lr = train_config.get("min_lr", 1e-6)
    base_lr = train_config.get("learning_rate", 1e-4)
    
    warmup_epochs = int(total_epochs * warmup_ratio)
    
    if scheduler_type == "cosine_warmup":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性 warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                # 确保不低于 min_lr / base_lr
                return max(min_lr / base_lr, cosine_decay)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    return scheduler