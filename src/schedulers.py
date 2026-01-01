# schedulers.py
"""
学习率调度器模块
- WarmupCosineScheduler: Warmup + Cosine Annealing
- 支持多种warmup策略
"""
import math
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim import Optimizer
from typing import Optional


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
) -> LambdaLR:
    """
    创建带warmup的余弦退火学习率调度器
    
    学习率变化：
    - Warmup阶段: 从0线性增长到base_lr
    - Cosine阶段: 从base_lr余弦退火到min_lr
    
    Args:
        optimizer: 优化器
        num_warmup_steps: warmup的步数（或epochs数）
        num_training_steps: 总训练步数（或epochs数）
        num_cycles: 余弦周期数，0.5表示半个周期（从1降到0）
        min_lr_ratio: 最小学习率与初始学习率的比值，默认0（降到0）
        last_epoch: 上一个epoch，用于恢复训练
        
    Returns:
        LambdaLR: 学习率调度器
        
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> # 100 epochs, 前10个epochs warmup
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, 
        ...     num_warmup_steps=10, 
        ...     num_training_steps=100
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup阶段：线性增长
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # 将cosine_decay从[0,1]映射到[min_lr_ratio, 1]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    创建带warmup的线性衰减学习率调度器
    
    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        num_training_steps: 总步数
        last_epoch: 上一个epoch
        
    Returns:
        LambdaLR: 学习率调度器
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, 
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class WarmupCosineScheduler(_LRScheduler):
    """
    带Warmup的余弦退火调度器（类实现版本）
    
    提供更多控制选项：
    - warmup_type: 'linear' 或 'exponential'
    - hold_epochs: warmup后保持峰值学习率的epochs数
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        warmup_type: str = 'linear',
        hold_epochs: int = 0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: warmup的epoch数
            total_epochs: 总epoch数
            min_lr: 最小学习率
            warmup_type: warmup类型，'linear' 或 'exponential'
            hold_epochs: warmup后保持峰值学习率的epoch数
            last_epoch: 上一个epoch
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_type = warmup_type
        self.hold_epochs = hold_epochs
        
        # 记录初始学习率
        self.base_lrs_record = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段
            if self.warmup_type == 'linear':
                alpha = self.last_epoch / max(1, self.warmup_epochs)
            else:  # exponential
                alpha = math.pow(self.last_epoch / max(1, self.warmup_epochs), 2)
            return [base_lr * alpha for base_lr in self.base_lrs_record]
        
        elif self.last_epoch < self.warmup_epochs + self.hold_epochs:
            # Hold阶段：保持峰值学习率
            return self.base_lrs_record
        
        else:
            # Cosine退火阶段
            decay_epochs = self.total_epochs - self.warmup_epochs - self.hold_epochs
            current_decay_epoch = self.last_epoch - self.warmup_epochs - self.hold_epochs
            
            if decay_epochs <= 0:
                return self.base_lrs_record
            
            progress = current_decay_epoch / decay_epochs
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay 
                for base_lr in self.base_lrs_record
            ]


# ============ 便捷函数 ============

def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    total_epochs: int,
    warmup_epochs: Optional[int] = None,
    warmup_ratio: float = 0.1,
    min_lr: float = 1e-7,
    **kwargs
) -> _LRScheduler:
    """
    创建学习率调度器的便捷函数
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
            - 'cosine_warmup': Warmup + Cosine (推荐用于对比学习)
            - 'linear_warmup': Warmup + Linear decay
            - 'cosine': 纯Cosine (无warmup)
        total_epochs: 总epoch数
        warmup_epochs: warmup的epoch数，如果不指定则使用warmup_ratio
        warmup_ratio: warmup占总epochs的比例，默认0.1 (10%)
        min_lr: 最小学习率
        **kwargs: 传递给具体调度器的额外参数
        
    Returns:
        学习率调度器
        
    Example:
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_type='cosine_warmup',
        ...     total_epochs=100,
        ...     warmup_ratio=0.1,  # 10 epochs warmup
        ...     min_lr=1e-6
        ... )
    """
    # 计算warmup epochs
    if warmup_epochs is None:
        warmup_epochs = int(total_epochs * warmup_ratio)
    
    if scheduler_type == 'cosine_warmup':
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_epochs,
            num_training_steps=total_epochs,
            min_lr_ratio=min_lr / optimizer.param_groups[0]['lr']
        )
    
    elif scheduler_type == 'linear_warmup':
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_epochs,
            num_training_steps=total_epochs
        )
    
    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
    
    elif scheduler_type == 'warmup_cosine_class':
        # 使用类实现版本，支持更多选项
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============ 可视化工具 ============

def plot_lr_schedule(
    scheduler: _LRScheduler,
    total_epochs: int,
    save_path: Optional[str] = None
):
    """
    可视化学习率调度曲线
    
    Args:
        scheduler: 学习率调度器
        total_epochs: 总epoch数
        save_path: 保存路径，如果为None则显示
    """
    import matplotlib.pyplot as plt
    
    lrs = []
    for epoch in range(total_epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(total_epochs), lrs, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, total_epochs])
    
    # 标注关键点
    max_lr = max(lrs)
    max_epoch = lrs.index(max_lr)
    ax.axvline(x=max_epoch, color='r', linestyle='--', alpha=0.5, label=f'Peak at epoch {max_epoch}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # 测试代码
    import torch
    
    # 创建一个dummy模型和优化器
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 测试 cosine_warmup
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='cosine_warmup',
        total_epochs=100,
        warmup_ratio=0.1,
        min_lr=1e-6
    )
    
    print("Learning rate schedule (cosine_warmup):")
    print(f"{'Epoch':>6} | {'LR':>12}")
    print("-" * 22)
    
    for epoch in range(100):
        lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0 or epoch < 15:
            print(f"{epoch:>6} | {lr:>12.2e}")
        scheduler.step()