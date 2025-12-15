import torch
from torch.utils.data import DataLoader
from ec import ec
# 从你的文件中导入需要测试的类和函数
# 确保 my_dataset.py 和这个脚本在同一个文件夹下
from dataset import triplet, collate_fn

# --- 请在这里配置你的文件路径和参数 ---
# 1. 你的CSV文件路径
CSV_FILE_PATH = "/home/fangchh/workdir/CLEAN/app/data/split10.csv"  # <- 修改这里

# 2. 你的距离矩阵pickle文件路径
DIST_MATRIX_PATH = "distance_dict.pkl" # <- 修改这里


# 4. 测试参数
BATCH_SIZE = 4
HARD_NEGATIVE_K = 2 # 这个值应该小于或等于你距离矩阵中每个ID的邻居数
# -----------------------------------------


def simple_test():
    """
    一个简单的测试函数，用于验证Dataset和DataLoader是否能正常工作。
    """
    print("--- 开始测试 ---")
    
    # 检查文件是否存在
    import os
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ 错误: 找不到CSV文件: {CSV_FILE_PATH}")
        return
    if not os.path.exists(DIST_MATRIX_PATH):
        print(f"❌ 错误: 找不到距离矩阵文件: {DIST_MATRIX_PATH}")
        return

    # 1. 初始化 Dataset
    print(f"[*] 正在从 '{CSV_FILE_PATH}' 初始化数据集...")
    try:
        dataset = triplet(
            csv_path=CSV_FILE_PATH,
            dist_path=DIST_MATRIX_PATH,
            hard_negative_k=HARD_NEGATIVE_K
        )
        print(f"✅ 数据集加载成功，共找到 {len(dataset)} 个样本。")
    except Exception as e:
        print(f"❌ 错误: 初始化Dataset时出错: {e}")
        return

    # 2. 初始化 DataLoader
    print(f"[*] 正在创建DataLoader，批次大小 (Batch Size) = {BATCH_SIZE}...")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True  # 通常训练时会打乱顺序
    )

    # 3. 尝试获取一个批次的数据
    print("[*] 正在尝试获取一个数据批次...")
    try:
        anchors, positives, negatives = next(iter(dataloader))
        print("✅ 成功获取一个数据批次！")
    except FileNotFoundError as e:
        print(f"❌ 错误: 获取数据时找不到文件。请检查 'ec' 目录是否存在，并且其中的 .pkl 文件名是否正确。")
        print(f"   具体错误: {e}")
        return
    except Exception as e:
        print(f"❌ 错误: 从DataLoader获取数据时出错: {e}")
        return

    # 4. 打印并验证输出信息
    print("\n--- 批次数据信息 ---")
    print(f"锚点 (Anchors) 的形状: {anchors}, 数据类型: {anchors.type()}")
    print(f"正样本 (Positives) 的形状: {positives}, 数据类型: {positives.type()}")
    print(f"负样本 (Negatives) 的形状: {negatives}, 数据类型: {negatives.type()}")

    # 基本断言检查
    assert anchors.shape[0] == BATCH_SIZE, "Anchor的批次大小不正确"
    assert positives.shape[0] == BATCH_SIZE, "Positive的批次大小不正确"
    assert negatives.shape[0] == BATCH_SIZE, "Negative的批次大小不正确"
    
    assert anchors.shape == positives.shape == negatives.shape, "三元组的形状不一致"
    
    print("\n--- 测试通过 ---")
    print("✅ 你的Dataset和DataLoader看起来工作正常！")


if __name__ == '__main__':
    simple_test()