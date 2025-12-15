import pandas as pd
from typing import Set, Dict

def get_ec_set_from_csv(csv_path: str) -> Set[str]:
    """
    从给定的CSV文件中读取数据，并返回一个包含所有不重复EC号的集合。

    Args:
        csv_path (str): CSV文件的路径。

    Returns:
        Set[str]: 一个包含所有唯一EC号的集合。
    """
    if not pd.io.common.file_exists(csv_path):
        print(f"警告：文件未找到于 '{csv_path}'，将返回空集合。")
        return set()
        
    try:
        df = pd.read_csv(csv_path, sep='\t')
        # 确保'EC number'列存在
        if 'EC number' not in df.columns:
            print(f"警告：文件 '{csv_path}' 中没有找到 'EC number' 列。")
            return set()
            
        # 分割、展开并获取所有唯一的EC号
        unique_ecs = df['EC number'].dropna().astype(str).str.split(';').explode().str.strip().unique()
        return set(unique_ecs)
    except Exception as e:
        print(f"读取或处理文件 '{csv_path}' 时出错: {e}")
        return set()

def analyze_overlap(train_ec_set: Set[str], test_ec_set: Set[str], test_set_name: str):
    """
    分析训练集和测试集之间的EC号重叠度，并打印详细报告。

    Args:
        train_ec_set (Set[str]): 训练集中的所有唯一EC号。
        test_ec_set (Set[str]): 测试集中的所有唯一EC号。
        test_set_name (str): 测试集的名称（用于打印）。
    """
    print(f"\n--- 正在分析 '{test_set_name}' 测试集 ---")
    
    if not test_ec_set:
        print("测试集为空或无法加载，无法进行分析。")
        return

    # 计算交集，即测试集中有多少EC号也在训练集中出现过
    intersection = train_ec_set.intersection(test_ec_set)
    
    # 计算测试集中有多少EC号是全新的（训练集中没有的）
    novel_ecs = test_ec_set.difference(train_ec_set)
    
    # 计算重叠百分比
    overlap_percentage = (len(intersection) / len(test_ec_set)) * 100 if test_ec_set else 0
    
    print(f"测试集中的EC号总数: {len(test_ec_set)}")
    print(f"与训练集重叠的EC号数量: {len(intersection)}")
    print(f"重叠度: {overlap_percentage:.2f}%")
    print(f"全新的EC号数量 (训练集中未见过): {len(novel_ecs)}")
    
    if novel_ecs:
        # 打印一些全新的EC号作为例子
        print(f"一些全新的EC号示例: {list(novel_ecs)[:5]}")
        
    if overlap_percentage == 0:
        print("\n结论：该测试集与训练集的标签空间完全不重叠。")
        print("KNN算法无法预测出任何训练集中不存在的标签，因此F1分数为0是符合预期的。")


if __name__ == "__main__":
    # --- 配置你的文件路径 ---
    # 主训练/验证/测试数据集的CSV文件
    MAIN_CSV_PATH = "./data/split100.csv"
    
    # 外部测试集的CSV文件
    EXTERNAL_TEST_SETS = {
        "PRICE": "./data/price.csv",
        "NEW": "./data/new.csv"
    }

    # 1. 从主数据集中获取训练集的EC号
    # 注意：我们应该使用整个主数据集的EC号作为参考，因为它代表了模型知识的来源
    print("正在加载训练集的EC号空间...")
    train_ecs = get_ec_set_from_csv(MAIN_CSV_PATH)
    print(f"从 '{MAIN_CSV_PATH}' 中加载了 {len(train_ecs)} 个唯一的EC号作为参考。")

    # 2. 逐一分析每个外部测试集
    for name, path in EXTERNAL_TEST_SETS.items():
        test_ecs = get_ec_set_from_csv(path)
        analyze_overlap(train_ecs, test_ecs, name)
