# create_csv_split.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import config as cfg
import os

# --- [学术配置] ---
# 目标分区索引 (0 = 第1折, 1 = 第2折, 以此类推)
# 您想要"第二折"，所以我们设置为 1
TARGET_FOLD_INDEX = 1 
# --------------------

# 1. 从 config.py 加载关键复现参数
SEED = cfg.SEED
N_SPLITS = cfg.N_SPLITS
CSV_PATH = cfg.DATA_PATHS["csv_path"]

# 2. 定义输出文件名
TRAIN_SPLIT_CSV = "train_split.csv"
VAL_SPLIT_CSV = "val_split.csv"

print(f"--- 正在创建固化的 CSV 划分 ---")
print(f"使用种子 (SEED): {SEED}")
print(f"总折数 (N_SPLITS): {N_SPLITS}")
print(f"目标分区索引 (TARGET_FOLD_INDEX): {TARGET_FOLD_INDEX}")
print(f"读取原始数据: {CSV_PATH}")


# 3. 加载完整的原始数据集
try:
    # 假设您的 CSV 至少有 'id' 和 'ec_number' 列
    # 您的 data_utils.load_and_create_label_map 暗示了这一点
    df_full = pd.read_csv(CSV_PATH, sep = '\t')
except FileNotFoundError:
    print(f"[错误] 找不到原始 CSV 文件: {CSV_PATH}")
    exit()
print(df_full.head)
# 4. 筛选出用于 K-Fold 划分的原始 ID
# (这复现了您在 main.py 中的逻辑)
original_ids_mask = ~df_full['Entry'].str.contains('_', na=False)
df_original = df_full[original_ids_mask].copy()
df_original_np_indices = df_original.index.to_numpy()

print(f"已加载 {len(df_full)} 行总数据。")
print(f"找到 {len(df_original)} 条用于 K-Fold 划分的原始 ID。")

# 5. 初始化 KFold 分割器，使用固定的 random_state
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# 6. 生成所有确定性的分割方案
all_splits = list(kf.split(df_original_np_indices))

# 7. 提取您指定的目标分区的 *索引*
print(f"正在提取分区 {TARGET_FOLD_INDEX} (即第 {TARGET_FOLD_INDEX + 1} 折)...")
train_original_indices, val_original_indices = all_splits[TARGET_FOLD_INDEX]

# 8. 获取真实的数据行索引 (这是最关键的一步)
train_split_indices = df_original_np_indices[train_original_indices]
val_split_indices = df_original_np_indices[val_original_indices]

# 9. 根据索引从原始 DataFrame 中选择行
df_train_split = df_full.loc[train_split_indices]
df_val_split = df_full.loc[val_split_indices]

print(f"\n--- 划分完成 ---")
print(f"训练集 (原始ID): {len(df_train_split)}")
print(f"验证集 (原始ID): {len(df_val_split)}")

# 10. 将这两个新的 DataFrames 保存为 CSV
df_train_split.to_csv(TRAIN_SPLIT_CSV, index=False)
df_val_split.to_csv(VAL_SPLIT_CSV, index=False)

print(f"\n[成功] 固定的数据划分已保存到：")
print(f"-> {os.path.abspath(TRAIN_SPLIT_CSV)}")
print(f"-> {os.path.abspath(VAL_SPLIT_CSV)}")