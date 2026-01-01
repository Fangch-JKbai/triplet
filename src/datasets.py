import random
from functools import lru_cache
import torch
from torch.utils.data import Dataset
from data_utils import _embedding_path, _has_embedding
from collections import defaultdict

# ==============================================================================
# --- ContrastiveDataset ---
# ==============================================================================
class ContrastiveDataset(Dataset):
    """
    为监督对比学习准备的数据集。
    - 在初始化时构建所有必要的映射表，增强了稳定性和复用性。
    - __getitem__ 直接返回整数形式的标签，简化了 collate_fn 的逻辑。
    - 预先构建了 ec_to_indices 映射，为PK采样器做好准备。
    """
    def __init__(self, ids, id_to_ecs_map, embedding_dir):
        self.embedding_dir = embedding_dir
        self.id_to_ecs_map = id_to_ecs_map
        self.ids = ids

        # <<< CHANGE 1: 在初始化时，一次性构建所有映射表 >>>
        print("Building EC mappings for ContrastiveDataset...")
        self.ec_to_int = {}
        self.int_to_ec = {}
        self.ec_to_indices = defaultdict(list)
        
        # 遍历所有ID，构建EC号的词汇表
        all_ecs = set()
        for ec_list in id_to_ecs_map.values():
            all_ecs.update(ec_list)
        
        # 排序以保证每次运行的映射顺序一致
        for ec in sorted(list(all_ecs)):
            if ec not in self.ec_to_int:
                new_int = len(self.ec_to_int)
                self.ec_to_int[ec] = new_int
                self.int_to_ec[new_int] = ec
        
        # 遍历数据集的ID，构建 ec -> [样本索引] 的映射
        for idx, seq_id in enumerate(self.ids):
            labels = self.id_to_ecs_map.get(seq_id, [])
            for ec in labels:
                if ec in self.ec_to_int: # 确保EC在我们的词汇表中
                    self.ec_to_indices[ec].append(idx)
        print(f"Mappings built. Found {len(self.ec_to_int)} unique EC numbers.")


    def __len__(self):
        return len(self.ids)

    @lru_cache(maxsize=200_000)
    def _load_embed(self, sample_id):
        t = torch.load(_embedding_path(self.embedding_dir, sample_id), map_location='cpu')
        return t['mean_representations'].float()

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        embedding = self._load_embed(seq_id)
        
        str_labels = self.id_to_ecs_map.get(seq_id, [])
        int_labels = [self.ec_to_int[ec] for ec in str_labels if ec in self.ec_to_int]
        
        return {"embedding_1280": embedding, "labels": int_labels}

# ==============================================================================
# --- contrastive_collate_fn (优化后) ---
# ==============================================================================
def contrastive_collate_fn(batch):
    """
    自定义 Collate Function (简化版)。
    - 它现在是无状态的，更加健壮。
    - 接收的已经是整数标签，只需随机选择一个并堆叠。
    """
    embeddings = []
    single_labels = []
    
    for item in batch:
        if item['labels']:
            embeddings.append(item['embedding_1280'])
            
            # 随机选择一个EC号作为此次的标签
            chosen_label = sorted(item['labels'])[0]
            single_labels.append(chosen_label)

    if not embeddings:
        return None # 如果批次为空，返回None

    return {
        "embeddings": torch.stack(embeddings),
        "labels": torch.tensor(single_labels, dtype=torch.long)
    }

# ==============================================================================
# --- EvaluationDataset (保持不变) ---
# EvaluationDataset 的设计已经非常优秀和鲁棒，无需修改。
# ==============================================================================
class EvaluationDataset(Dataset):
    def __init__(self, ids, id_to_ecs_map, embedding_dir, strict_embeddings=True):
        self.embedding_dir = embedding_dir
        self.id_to_ecs_map = id_to_ecs_map
        self.strict_embeddings = strict_embeddings

        if strict_embeddings:
            filtered, missing = [], []
            for sid in ids:
                if _has_embedding(embedding_dir, sid):
                    filtered.append(sid)
                else:
                    missing.append(sid)
            if missing:
                print(f"[Eval] {len(missing)} ids without embeddings are skipped (strict mode).")
            self.ids = filtered
        else:
            self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    @lru_cache(maxsize=200_000)
    def _load_embed(self, sample_id):
        t = torch.load(_embedding_path(self.embedding_dir, sample_id), map_location='cpu')
        return t['mean_representations'].float()

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        labels = self.id_to_ecs_map.get(seq_id.split('_')[0], [])
        if not labels:
            labels = self.id_to_ecs_map.get(seq_id, [])

        if not self.strict_embeddings and not _has_embedding(self.embedding_dir, seq_id):
            return {"skip": True}

        embedding = self._load_embed(seq_id)
        return {"embedding_1280": embedding, "labels": labels, "seq_id": seq_id, "skip": False}