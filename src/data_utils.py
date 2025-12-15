# data_utils.py
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Set

# --- Embedding 文件相关的辅助函数 ---
def _embedding_path(embedding_dir: str, sample_id: str) -> str:
    return os.path.join(embedding_dir, f"{sample_id}.pt")

def _has_embedding(embedding_dir: str, sample_id: str) -> bool:
    return os.path.exists(_embedding_path(embedding_dir, sample_id))

# --- 数据映射与过滤相关的函数 ---
def load_and_create_label_map(csv_path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    更稳健的读取：自动识别分隔符，兼容 NaN，过滤空标签 (优化了字典创建过程)。
    """
    if not os.path.exists(csv_path):
        return {}, []
    df = pd.read_csv(csv_path, sep=None, engine='python')
    required = {'Entry', 'EC number'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} 缺少必要列：{required - set(df.columns)}")

    df['EC number'] = df['EC number'].fillna('').astype(str)
    df['EC number'] = df['EC number'].apply(lambda s: [t.strip() for t in s.split(';') if t.strip()])
    df = df[df['EC number'].map(len) > 0].reset_index(drop=True)

    id_to_ecs = dict(zip(df['Entry'], df['EC number']))
    
    return id_to_ecs, list(id_to_ecs.keys())

def augment_map_with_mutants(
    id_to_ecs_master: Dict[str, List[str]], 
    embedding_dir: str, 
    train_ids_orig_only: List[str] = None
) -> Dict[str, List[str]]:
    """
    让“训练原始ID”的变体”继承原始 ID 的标签 (增加了可读性和进度条)。
    """
    files = [f for f in os.listdir(embedding_dir) if f.endswith('.pt')]
    mutant_ids = [f[:-3] for f in files if '_' in f]
    add = {}
    train_orig_set = set(train_ids_orig_only) if train_ids_orig_only is not None else None

    for mid in tqdm(mutant_ids, desc="Augmenting with mutants"):
        base = mid.split('_')[0]
        if base in id_to_ecs_master:
            if (train_orig_set is None) or (base in train_orig_set):
                add[mid] = id_to_ecs_master[base]

    if add:
        print(f"Augmenting label map with {len(add)} mutant IDs.")
        id_to_ecs_master.update(add)
    return id_to_ecs_master

def build_ec_to_ids(id_to_ecs_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    构建 EC -> [ID列表] 的反向映射 (使用defaultdict重写，性能大幅提升)。
    """
    ec_to_ids = defaultdict(list)
    for entry, ecs in id_to_ecs_map.items():
        for ec in ecs:
            ec_to_ids[ec].append(entry)
    return dict(ec_to_ids)

def filter_ids_with_embeddings(ids: List[str], embedding_dir: str, msg_prefix: str = "") -> List[str]:
    """
    过滤出拥有嵌入向量文件的ID列表 (优化了文件检查逻辑，性能大幅提升)。
    """
    try:
        available_embeddings: Set[str] = {f.split('.')[0] for f in os.listdir(embedding_dir)}
    except FileNotFoundError:
        print(f"Warning: Embedding directory not found at {embedding_dir}. Returning empty list.")
        return []

    ok, miss = [], []
    for sid in tqdm(ids, desc=f"{msg_prefix}Filtering IDs"):
        if sid in available_embeddings:
            ok.append(sid)
        else:
            miss.append(sid)
            
    if miss:
        print(f"{msg_prefix}Found {len(miss)} IDs without embeddings; they are skipped.")
    return ok