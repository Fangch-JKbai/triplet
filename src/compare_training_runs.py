"""
对比多次训练运行的性能曲线
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import argparse

def load_history(history_path):
    """加载训练历史"""
    with open(history_path, 'r') as f:
        return json.load(f)

def plot_comparison(histories, labels, output_path='comparison.png'):
    """
    对比绘制多次训练的曲线

    Args:
        histories: list of dict, 每个dict是一个训练历史
        labels: list of str, 每次训练的标签
        output_path: 输出图片路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']

    # 1. Price AUC对比
    ax1 = axes[0, 0]
    for i, (history, label) in enumerate(zip(histories, labels)):
        if 'price_auc' in history and history['price_auc']:
            epochs = history['epoch']
            auc = history['price_auc']
            ax1.plot(epochs, auc,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    label=label,
                    linewidth=2,
                    markersize=4,
                    alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('Price Dataset AUC Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. New AUC对比
    ax2 = axes[0, 1]
    for i, (history, label) in enumerate(zip(histories, labels)):
        if 'new_auc' in history and history['new_auc']:
            epochs = history['epoch']
            auc = history['new_auc']
            ax2.plot(epochs, auc,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    label=label,
                    linewidth=2,
                    markersize=4,
                    alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('New Dataset AUC Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 平均AUC对比
    ax3 = axes[1, 0]
    for i, (history, label) in enumerate(zip(histories, labels)):
        if 'price_auc' in history and 'new_auc' in history:
            if history['price_auc'] and history['new_auc']:
                epochs = history['epoch']
                avg_auc = [(p + n) / 2 for p, n in zip(history['price_auc'], history['new_auc'])]
                ax3.plot(epochs, avg_auc,
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        label=label,
                        linewidth=2,
                        markersize=4,
                        alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Average AUC', fontsize=12)
    ax3.set_title('Average Test AUC Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. 训练损失对比
    ax4 = axes[1, 1]
    for i, (history, label) in enumerate(zip(histories, labels)):
        if 'train_loss' in history and history['train_loss']:
            epochs = list(range(1, len(history['train_loss']) + 1))
            train_loss = history['train_loss']
            ax4.plot(epochs, train_loss,
                    color=colors[i % len(colors)],
                    label=label,
                    linewidth=2,
                    alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Training Loss', fontsize=12)
    ax4.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison plot saved to: {output_path}")

def print_summary(histories, labels):
    """打印对比摘要"""
    print("\n" + "="*80)
    print("TRAINING RUNS COMPARISON SUMMARY")
    print("="*80)

    for label, history in zip(labels, histories):
        print(f"\n{label}:")
        print("-" * 60)

        if 'price_auc' in history and history['price_auc']:
            price_aucs = history['price_auc']
            max_price_auc = max(price_aucs)
            max_price_idx = price_aucs.index(max_price_auc)
            max_price_epoch = history['epoch'][max_price_idx]
            print(f"  Price AUC:")
            print(f"    Best: {max_price_auc:.4f} at epoch {max_price_epoch}")
            print(f"    Final: {price_aucs[-1]:.4f} at epoch {history['epoch'][-1]}")

        if 'new_auc' in history and history['new_auc']:
            new_aucs = history['new_auc']
            max_new_auc = max(new_aucs)
            max_new_idx = new_aucs.index(max_new_auc)
            max_new_epoch = history['epoch'][max_new_idx]
            print(f"  New AUC:")
            print(f"    Best: {max_new_auc:.4f} at epoch {max_new_epoch}")
            print(f"    Final: {new_aucs[-1]:.4f} at epoch {history['epoch'][-1]}")

        if 'price_auc' in history and 'new_auc' in history:
            if history['price_auc'] and history['new_auc']:
                avg_aucs = [(p + n) / 2 for p, n in zip(history['price_auc'], history['new_auc'])]
                max_avg_auc = max(avg_aucs)
                max_avg_idx = avg_aucs.index(max_avg_auc)
                max_avg_epoch = history['epoch'][max_avg_idx]
                print(f"  Average AUC:")
                print(f"    Best: {max_avg_auc:.4f} at epoch {max_avg_epoch}")
                print(f"    Final: {avg_aucs[-1]:.4f} at epoch {history['epoch'][-1]}")

        if 'train_loss' in history and history['train_loss']:
            final_loss = history['train_loss'][-1]
            print(f"  Final Training Loss: {final_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Compare multiple training runs')
    parser.add_argument('--runs', nargs='+', required=True,
                       help='List of experiment directories to compare')
    parser.add_argument('--labels', nargs='+',
                       help='Labels for each run (optional)')
    parser.add_argument('--output', default='training_comparison.png',
                       help='Output plot filename')

    args = parser.parse_args()

    # 加载所有历史
    histories = []
    labels = args.labels if args.labels else []

    for i, run_dir in enumerate(args.runs):
        history_path = os.path.join(run_dir, 'training_history.json')

        if not os.path.exists(history_path):
            print(f"Warning: {history_path} not found, skipping...")
            continue

        history = load_history(history_path)
        histories.append(history)

        # 如果没有提供label，使用目录名
        if i >= len(labels):
            labels.append(os.path.basename(run_dir))

    if not histories:
        print("Error: No valid training histories found!")
        return

    print(f"\nComparing {len(histories)} training runs:")
    for label in labels:
        print(f"  - {label}")

    # 绘制对比图
    plot_comparison(histories, labels, args.output)

    # 打印摘要
    print_summary(histories, labels)

    print("\n" + "="*80)
    print("Comparison completed!")
    print("="*80)

if __name__ == '__main__':
    # 如果不带参数运行，自动查找最近的实验
    import sys

    if len(sys.argv) == 1:
        print("Usage examples:")
        print()
        print("# 对比两次实验")
        print("python compare_training_runs.py \\")
        print("  --runs experiments/run1 experiments/run2 \\")
        print("  --labels 'Base LR=1e-3' 'Base LR=1e-4'")
        print()
        print("# 自动查找最近的3次实验")
        print("python compare_training_runs.py \\")
        print("  --runs experiments/train_monitored_2024-* \\")
        print("  --output my_comparison.png")
        print()

        # 尝试自动查找
        exp_dirs = sorted(glob('experiments/train_monitored_*'), key=os.path.getmtime, reverse=True)
        if exp_dirs:
            print(f"Found {len(exp_dirs)} recent experiments:")
            for i, exp_dir in enumerate(exp_dirs[:5], 1):
                print(f"  {i}. {exp_dir}")
            print()
            print("Tip: Copy one of the commands above and modify the run directories")
    else:
        main()
