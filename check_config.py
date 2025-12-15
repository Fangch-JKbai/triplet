#!/usr/bin/env python3
"""
配置检查脚本 - 验证所有路径和参数是否正确
"""

import os
import sys
import torch

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_file_exists(path, description):
    """检查文件是否存在"""
    if os.path.exists(path):
        print(f"  ✓ {description}: {path}")
        return True
    else:
        print(f"  ✗ {description} 不存在: {path}")
        return False

def check_directory_exists(path, description):
    """检查目录是否存在"""
    if os.path.exists(path) and os.path.isdir(path):
        # 统计目录中的文件数
        try:
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"  ✓ {description}: {path} (包含 {file_count} 个文件)")
        except:
            print(f"  ✓ {description}: {path}")
        return True
    else:
        print(f"  ✗ {description} 不存在: {path}")
        return False

def main():
    print("="*60)
    print("配置检查")
    print("="*60)
    print()

    # 导入配置
    try:
        import config as cfg
        print("✓ 成功加载 config.py")
    except Exception as e:
        print(f"✗ 无法加载 config.py: {e}")
        return False

    print()
    all_ok = True

    # 1. 检查设备配置
    print("1. 设备配置")
    print(f"  配置的设备: {cfg.DEVICE}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA 可用")
        print(f"    - GPU数量: {torch.cuda.device_count()}")
        print(f"    - 当前GPU: {torch.cuda.current_device()}")
        if hasattr(torch.cuda, 'get_device_name'):
            print(f"    - GPU名称: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠ CUDA 不可用，将使用CPU")
    print()

    # 2. 检查实验模式
    print("2. 实验模式")
    print(f"  当前模式: {cfg.EXPERIMENT_MODE}")
    if cfg.EXPERIMENT_MODE not in ["hyperparameter_search", "final_training", "eval_hyperparam_search"]:
        print(f"  ⚠ 警告: 未知的实验模式")
    print()

    # 3. 检查K折配置
    print("3. 交叉验证配置")
    print(f"  K折数: {cfg.N_SPLITS}")
    print(f"  随机种子: {cfg.SEED}")
    print()

    # 4. 检查数据路径
    print("4. 数据路径检查")

    # 主数据集
    csv_path = cfg.DATA_PATHS.get("csv_path", "")
    if not check_file_exists(csv_path, "主数据集CSV"):
        all_ok = False
        print(f"    提示: 请修改 config.py 中的 DATA_PATHS['csv_path']")

    # Embedding目录
    emb_dir = cfg.DATA_PATHS.get("embedding_dir", "")
    if not check_directory_exists(emb_dir, "Embedding目录"):
        all_ok = False
        print(f"    提示: 请修改 config.py 中的 DATA_PATHS['embedding_dir']")

    # 外部测试集
    ext_tests = cfg.DATA_PATHS.get("external_test_sets", {})
    for name, path in ext_tests.items():
        if not check_file_exists(path, f"测试集 {name}"):
            all_ok = False
            print(f"    提示: 请修改 config.py 中的 DATA_PATHS['external_test_sets']['{name}']")

    print()

    # 5. 检查训练配置
    print("5. 训练配置")
    train_cfg = cfg.TRAIN_CONFIG
    print(f"  Epoch数: {train_cfg.get('epochs', 'N/A')}")
    print(f"  学习率: {train_cfg.get('learning_rate', 'N/A')}")
    print(f"  Early Stopping Patience: {train_cfg.get('patience', 'N/A')}")
    print(f"  Temperature: {train_cfg.get('temperature', 'N/A')}")
    print(f"  P (类别数/batch): {train_cfg.get('P', 'N/A')}")
    print(f"  K (每类样本数): {train_cfg.get('K', 'N/A')}")
    print(f"  Batch Size: {train_cfg.get('P', 0) * train_cfg.get('K', 0)}")
    print()

    # 6. 检查模型配置
    print("6. 模型配置")
    model_cfg = cfg.MODEL_CONFIG
    print(f"  输入维度: {model_cfg.get('input_dim', 'N/A')}")
    print(f"  嵌入维度 d_z: {model_cfg.get('d_z', 'N/A')}")
    print(f"  Dropout率: {model_cfg.get('dropout_rate', 'N/A')}")
    print()

    # 7. 检查评估配置
    print("7. 评估配置")
    eval_cfg = cfg.EVAL_CONFIG
    print(f"  默认 k: {eval_cfg.get('default_k', 'N/A')}")
    print(f"  默认 tau: {eval_cfg.get('default_tau', 'N/A')}")
    print(f"  加权评估: {eval_cfg.get('weighted', 'N/A')}")
    print()

    # 8. 检查输出目录
    print("8. 输出目录")
    output_dir = cfg.ARTIFACTS_CONFIG.get('output_dir', 'experiments')
    os.makedirs(output_dir, exist_ok=True)
    print(f"  ✓ 输出目录: {output_dir}")
    print()

    # 9. 检查关键Python包
    print("9. Python依赖包")
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
    ]

    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} 未安装")
            all_ok = False

    print()
    print("="*60)

    if all_ok:
        print("✓ 所有检查通过！可以开始训练。")
        print()
        print("推荐的运行方式：")
        print("  1. 五折交叉验证: cd src && python main_kfold.py")
        print("  2. 最终训练: cd src && python main.py")
        print("  3. 或使用自动化脚本: ./run_complete_workflow.sh")
    else:
        print("✗ 发现配置问题，请修复后再运行训练。")
        return False

    print("="*60)
    return all_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
