import os
import torch
import csv
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 导入你的配置和工具
import config as cfg
import data_utils
from datasets import EvaluationDataset

# ================= 配置区域 =================
OUTPUT_DIR = "visualization_data-base"
MAX_SAMPLES = 5000
SEED = 42

# ✅ 新增：只挑选大约 8 个 EC（四位）种类
N_EC_TYPES = 8

# 选择策略：
# - "topk": 选择样本数最多的前 N_EC_TYPES 个 EC（推荐，保证每类够多）
# - "random": 随机挑 N_EC_TYPES 个 EC
EC_SELECT_MODE = "topk"

# 每个 EC 最多采样多少（防止某个 EC 过多挤占）
# 如果你希望每类尽量均衡，可以保留；如果想完全按 MAX_SAMPLES 均分，也可以设为 None
PER_EC_CAP = None  # 例如 800；None 表示不单独限制

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def export_cpt_embeddings():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🚀 开始准备数据...")
    print(f"数据源 CSV: {cfg.DATA_PATHS['csv_path']}")

    # 这里强制使用 CPT 的 embedding 路径（你原注释里说 CPT，但这里用的是 cfg.embedding_paths["esm"]）
    cpt_embedding_dir = cfg.embedding_paths["esm"]
    print(f"Embedding 路径 (CPT): {cpt_embedding_dir}")

    # 1) 加载全量映射
    id_to_ecs_master, _ = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])

    # 2) 按“四位EC”聚合单标签样本
    ec_to_ids = defaultdict(list)
    print("正在筛选单标签并按 EC(四位) 聚合...")
    for seq_id, ec_list in id_to_ecs_master.items():
        if len(ec_list) != 1:
            continue
        ec_full = ec_list[0]  # e.g. "1.1.1.1"
        if not isinstance(ec_full, str) or ec_full.count(".") < 3:
            continue
        ec_to_ids[ec_full].append(seq_id)

    all_single = sum(len(v) for v in ec_to_ids.values())
    print(f"原始单标签样本总数: {all_single}")
    print(f"单标签 EC(四位) 种类数: {len(ec_to_ids)}")

    if len(ec_to_ids) == 0:
        raise RuntimeError("没有找到任何单标签四位 EC 样本，请检查 csv 映射是否正确。")

    # 3) 选择 ~8 个 EC 种类
    ec_types = list(ec_to_ids.keys())
    if EC_SELECT_MODE == "topk":
        ec_types = sorted(ec_types, key=lambda ec: len(ec_to_ids[ec]), reverse=True)
        chosen_ecs = ec_types[:min(N_EC_TYPES, len(ec_types))]
    elif EC_SELECT_MODE == "random":
        chosen_ecs = random.sample(ec_types, k=min(N_EC_TYPES, len(ec_types)))
    else:
        raise ValueError(f"EC_SELECT_MODE 不支持: {EC_SELECT_MODE}")

    print("\n✅ 选中的 EC(四位) 种类：")
    for ec in chosen_ecs:
        print(f"  - {ec}: {len(ec_to_ids[ec])} samples")

    # 4) 从这 8 个 EC 里采样（尽量均衡）
    # 先算每类目标采样数
    n_types = len(chosen_ecs)
    target_total = min(MAX_SAMPLES, sum(len(ec_to_ids[ec]) for ec in chosen_ecs))
    base_per_ec = max(1, target_total // n_types)

    if PER_EC_CAP is not None:
        base_per_ec = min(base_per_ec, PER_EC_CAP)

    selected_ids = []
    remaining_pool = {ec: [] for ec in chosen_ecs}

    # 4.1 先每类取 base_per_ec
    for ec in chosen_ecs:
        ids = ec_to_ids[ec][:]
        random.shuffle(ids)
        take = min(base_per_ec, len(ids))
        selected_ids.extend(ids[:take])
        remaining_pool[ec] = ids[take:]

    # 4.2 如果还没到 target_total，就从剩余里“轮询补齐”
    need = target_total - len(selected_ids)
    if need > 0:
        ec_cycle = chosen_ecs[:]
        idx = 0
        while need > 0 and any(len(remaining_pool[ec]) > 0 for ec in ec_cycle):
            ec = ec_cycle[idx % len(ec_cycle)]
            if remaining_pool[ec]:
                selected_ids.append(remaining_pool[ec].pop())
                need -= 1
            idx += 1

    # 如果还是超过 MAX_SAMPLES（理论上不会），截断一下
    if len(selected_ids) > MAX_SAMPLES:
        selected_ids = selected_ids[:MAX_SAMPLES]

    print(f"\n最终用于可视化的样本数: {len(selected_ids)}")
    print(f"覆盖 EC(四位) 种类数: {len(set(id_to_ecs_master[i][0] for i in selected_ids if i in id_to_ecs_master and len(id_to_ecs_master[i])==1))}")

    # 5) 构建 Dataset
    dataset = EvaluationDataset(
        ids=selected_ids,
        id_to_ecs_map=id_to_ecs_master,
        embedding_dir=cpt_embedding_dir,
        strict_embeddings=True
    )

    # 6) 导出文件
    vec_path = os.path.join(OUTPUT_DIR, "vectors.tsv")
    meta_path = os.path.join(OUTPUT_DIR, "metadata.tsv")

    vectors = []
    metadatas = []

    print("开始提取特征和标签...")
    for i in tqdm(range(len(dataset))):
        if len(vectors) >= MAX_SAMPLES:   # ✅ 最终写入也硬限制 5000
            break

        sample = dataset[i]
        if sample.get("skip", False):
            continue

        emb = sample["embedding_1280"].numpy()
        ec_list = sample["labels"]
        if not ec_list:
            continue
        ec_full = ec_list[0]

        vectors.append(emb)
        metadatas.append([sample["seq_id"], ec_full])


    print(f"正在写入 {vec_path} ...")
    with open(vec_path, "w", encoding="utf-8") as f:
        for vec in vectors:
            f.write("\t".join([f"{x:.6f}" for x in vec]) + "\n")

    print(f"正在写入 {meta_path} ...")
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["SeqID", "EC_Number"])
        writer.writerows(metadatas)

    print("\n" + "=" * 50)
    print("✅ 处理完成！")
    print(f"1. 向量文件: {os.path.abspath(vec_path)}")
    print(f"2. 标签文件: {os.path.abspath(meta_path)}")
    print("=" * 50)
    print("下一步操作:")
    print("1. 打开 https://projector.tensorflow.org/")
    print("2. 点击 Load，上传 vectors.tsv 和 metadata.tsv")
    print("3. 左侧 'Color by' 选择 'EC_Number'（现在是四位 EC）")
    print("=" * 50)

if __name__ == "__main__":
    export_cpt_embeddings()
