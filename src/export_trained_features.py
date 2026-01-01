import os
import torch
import csv
import random
import numpy as np
from tqdm import tqdm

# 导入你的配置、工具和模型定义
import config as cfg
import data_utils
from datasets import EvaluationDataset
# 假设你的模型定义在 model.py 里，类名叫 EcClassifier
from main import EcClassifier # 请根据实际情况修改导入路径

# ================= 配置区域 =================
# 1. 选择模式和对应的模型权重路径
# MODE = "BASE"
# MODEL_WEIGHTS_PATH = "experiments/base_run/checkpoints/best_checkpoint.pt" # 修改为你Base模型的路径

MODE = "CPT"
# 修改为你 CPT 模型训练好的权重路径 (例如 Epoch 2 的那个巅峰模型)
MODEL_WEIGHTS_PATH = "/home/fangchh/workdir/triplet/src/experiments/cpt/best_model.pth" 

# 2. 输出目录
OUTPUT_DIR = f"visualization_{MODE.lower()}_features"

# 3. 采样上限 (建议 5000)
MAX_SAMPLES = 5000
SEED = 42
DEVICE = cfg.DEVICE
# ============================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_trained_model(weights_path, device):
    """加载训练好的模型权重"""
    print(f"正在加载模型权重: {weights_path}")
    # 初始化模型结构
    model = EcClassifier(**cfg.MODEL_CONFIG).to(device)
    
    # 加载权重
    # 注意：根据你保存 checkpoint 的方式，可能需要调整这里的键名
    # 比如有些是 checkpoint['model_state_dict']，有些直接是 state_dict
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit(1)
        
    model.eval() # 一定要设置为评估模式
    return model

def export_features():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 开始导出 [{MODE}] 模式的 Head 特征...")
    
    # 1. 加载模型
    model = load_trained_model(MODEL_WEIGHTS_PATH, DEVICE)
    
    raw_embedding_dir = cfg.embedding_paths["cpt"]
    print(f"Raw Embedding 来源: {raw_embedding_dir}")

    id_to_ecs_master, _ = data_utils.load_and_create_label_map(cfg.DATA_PATHS["csv_path"])

    # 筛选单标签数据
    valid_ids = [seq_id for seq_id, ec_list in id_to_ecs_master.items() if len(ec_list) == 1]
    print(f"原始单标签样本总数: {len(valid_ids)}")

    # 随机采样
    if len(valid_ids) > MAX_SAMPLES:
        random.shuffle(valid_ids)
        selected_ids = valid_ids[:MAX_SAMPLES]
    else:
        selected_ids = valid_ids
    print(f"采样样本数: {len(selected_ids)}")

    # 构建 Dataset
    dataset = EvaluationDataset(
        ids=selected_ids,
        id_to_ecs_map=id_to_ecs_master,
        embedding_dir=raw_embedding_dir,
        strict_embeddings=True
    )

    # 3. 提取特征
    vectors = []
    metadatas = []

    print("开始提取 Head 特征...")
    with torch.no_grad(): # 关键：不计算梯度
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            if sample.get('skip', False): continue
            
            # 获取原始 1280 维 embedding 并移动到设备
            raw_emb = sample['embedding_1280'].to(DEVICE)
            
            # --- 核心步骤：通过模型 Head ---
            feature_128 = model(raw_emb)
            
            # 转回 CPU numpy
            vec = feature_128.cpu().numpy()
            
            # 处理元数据
            ec_list = sample['labels']
            if not ec_list: continue
            ec_full = ec_list[0]
            try:
                main_class = ec_full.split('.')[0]
            except:
                main_class = "Unknown"

            vectors.append(vec)
            metadatas.append([sample['seq_id'], ec_full, main_class])

    # 4. 写入文件
    vec_path = os.path.join(OUTPUT_DIR, 'vectors.tsv')
    meta_path = os.path.join(OUTPUT_DIR, 'metadata.tsv')

    print(f"正在写入向量 (维度: {len(vectors[0])})...")
    with open(vec_path, 'w', encoding='utf-8') as f:
        for vec in vectors:
            f.write('\t'.join([f"{x:.6f}" for x in vec]) + '\n')

    print("正在写入元数据...")
    with open(meta_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['SeqID', 'EC_Number', 'Main_Class'])
        writer.writerows(metadatas)

    print("\n" + "="*50)
    print(f"✅ [{MODE}] 特征导出完成！")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print("="*50)

if __name__ == '__main__':
    export_features()