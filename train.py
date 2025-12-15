import os
import random
import pickle
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

### 修改点 ###
# 引入 roc_auc_score 用于计算 AUC
from sklearn.metrics import roc_auc_score
# ########### #

# -----------------------------
# Set Random Seeds for Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------------
# 1. Model Definition
# -----------------------------
class EcClassifier(nn.Module):
    def __init__(self, input_dim=1280, d1=512, d2=256, d_z=128, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d1)
        self.ln1 = nn.LayerNorm(d1)
        self.fc2 = nn.Linear(d1, d2)
        self.ln2 = nn.LayerNorm(d2)
        self.fc3 = nn.Linear(d2, d_z)
        self.dropout = nn.Dropout(dropout_rate)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = F.relu(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.ln2(self.fc2(h)))
        h = self.dropout(h)
        z = self.fc3(h)
        return F.normalize(z, p=2, dim=-1)  # 与余弦距离/相似度天然匹配


# -----------------------------
# 2. Dataset Definitions
# -----------------------------
def _embedding_path(embedding_dir, sample_id):
    return os.path.join(embedding_dir, f"{sample_id}.pt")


def _has_embedding(embedding_dir, sample_id):
    return os.path.exists(_embedding_path(embedding_dir, sample_id))


class TripletDataset(Dataset):
    """
    - 训练集使用：原始训练ID + 其变体（仅来源于训练原始ID，避免泄漏）
    - 为了减少 I/O 抖动，加入 LRU 内存缓存
    """
    def __init__(self, train_ids, id_to_ecs_map, ec_to_ids_map, dist_matrix,
                 embedding_dir, hard_negative_k=5, missing_policy="skip"):
        super().__init__()
        self.train_ids = train_ids
        self.id_to_ecs_map = id_to_ecs_map
        self.ec_to_ids_map = ec_to_ids_map
        self.dist_matrix = dist_matrix
        self.embedding_dir = embedding_dir
        self.hard_negative_k = hard_negative_k
        self.missing_policy = missing_policy  # "skip" or "error"
        self._precompute_hard_negatives()
        self.triplets = []
        self.resample_triplets()

    def _precompute_hard_negatives(self):
        print("Pre-computing hard negative EC lookups...")
        self.ec_hard_negatives = {}
        for ec in self.ec_to_ids_map.keys():
            neighbor_ecs = self.dist_matrix.get(ec, {})
            if isinstance(neighbor_ecs, dict):
                distances = sorted(neighbor_ecs.items(), key=lambda x: x[1])
                self.ec_hard_negatives[ec] = [d[0] for d in distances[:self.hard_negative_k]]
            else:
                self.ec_hard_negatives[ec] = []

    def resample_triplets(self):
        print("\nResampling triplets for the new epoch...")
        new_triplets = []
        train_ids_set = set(self.train_ids)

        skipped_missing = 0
        for a_id in tqdm(self.train_ids, desc="Generating New Triplets"):
            anchor_ec_list = self.id_to_ecs_map.get(a_id, [])
            if not anchor_ec_list:
                continue

            for chosen_anchor_ec in anchor_ec_list:
                # 正样本候选：同一 EC 的所有样本（包含变体），且在训练池中
                positive_candidates = [
                    pid for pid in self.ec_to_ids_map.get(chosen_anchor_ec, [])
                    if pid != a_id and pid in train_ids_set
                ]
                if not positive_candidates:
                    continue

                # 硬负样本候选：来自“最相近的若干 EC”的样本（包含变体）
                hard_negative_ecs = self.ec_hard_negatives.get(chosen_anchor_ec, [])
                negative_candidates = []
                for neg_ec in hard_negative_ecs:
                    cands = [
                        nid for nid in self.ec_to_ids_map.get(neg_ec, [])
                        if nid in train_ids_set
                    ]
                    negative_candidates.extend(cands)

                if not negative_candidates:
                    continue

                p_id = random.choice(positive_candidates)
                n_id = random.choice(negative_candidates)

                # 若缺嵌入，则根据 missing_policy 处理
                if not (_has_embedding(self.embedding_dir, a_id) and
                        _has_embedding(self.embedding_dir, p_id) and
                        _has_embedding(self.embedding_dir, n_id)):
                    if self.missing_policy == "skip":
                        skipped_missing += 1
                        continue
                    else:
                        raise FileNotFoundError("Triplet contains id(s) without embedding .pt")

                new_triplets.append((a_id, p_id, n_id))

        self.triplets = new_triplets
        print(f"Generated {len(self.triplets)} new triplets for this epoch. "
              f"(skipped {skipped_missing} triplets due to missing embeddings)")

    def __len__(self):
        return len(self.triplets)

    @lru_cache(maxsize=200_000)
    def _load_embed(self, sample_id):
        t = torch.load(_embedding_path(self.embedding_dir, sample_id), map_location='cpu')
        return t['mean_representations'].float()

    def __getitem__(self, idx):
        a_id, p_id, n_id = self.triplets[idx]
        a_embed = self._load_embed(a_id)
        p_embed = self._load_embed(p_id)
        n_embed = self._load_embed(n_id)
        return {"anchor": a_embed, "positive": p_embed, "negative": n_embed}


class EvaluationDataset(Dataset):
    """
    strict_embeddings=True  时：在 __init__ 预先过滤没有嵌入的 ID（用于 gallery/val/test 主流程）
    strict_embeddings=False 时：不做预过滤；__getitem__ 若缺嵌入，返回 {'skip': True} 交给上层跳过（用于外部测试集“无过滤”需求）
    """
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
            # 不做任何预过滤
            self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    @lru_cache(maxsize=200_000)
    def _load_embed(self, sample_id):
        t = torch.load(_embedding_path(self.embedding_dir, sample_id), map_location='cpu')
        return t['mean_representations'].float()

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        # 标签优先取原始ID（变体继承时已加入 map，这里仍做保险）
        labels = self.id_to_ecs_map.get(seq_id.split('_')[0], [])
        if not labels:
            labels = self.id_to_ecs_map.get(seq_id, [])

        # strict=False: 不预过滤，运行时尝试加载；缺失则返回 skip
        if not self.strict_embeddings and not _has_embedding(self.embedding_dir, seq_id):
            return {"skip": True}

        embedding = self._load_embed(seq_id)
        return {"embedding_1280": embedding, "labels": labels, "skip": False}


# -----------------------------
# 3. Evaluation Utilities
# -----------------------------
def evaluation_collate_fn(batch):
    """支持 batch 中包含 {'skip': True} 的样本：这些样本在此被跳过，不会引发崩溃"""
    valid = [b for b in batch if not (isinstance(b, dict) and b.get("skip", False))]
    if len(valid) == 0:
        # 返回一个占位，后续逻辑会识别到空 batch 并跳过
        return {"empty": True}
    embedding_tensors = torch.stack([item['embedding_1280'] for item in valid])
    label_lists = [item['labels'] for item in valid]
    return {"embedding_1280": embedding_tensors, "labels": label_lists, "empty": False}


def get_all_embeddings_and_labels(classifier, dataloader, device):
    classifier.eval()
    all_embeddings_128, all_labels = [], []
    empty_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings for evaluation"):
            if isinstance(batch, dict) and batch.get("empty", False):
                empty_batches += 1
                continue
            embeddings_1280 = batch['embedding_1280'].to(device, non_blocking=True)
            with autocast(enabled=True):
                embeddings_128 = classifier(embeddings_1280)
            all_embeddings_128.append(embeddings_128.cpu())
            all_labels.extend(batch['labels'])
    if empty_batches > 0:
        print(f"[Eval] {empty_batches} empty batches (caused by missing embeddings) were skipped at runtime.")
    if len(all_labels) == 0:
        # 返回空张量以避免后续计算报错
        return torch.empty((0, 128)), []
    return torch.cat(all_embeddings_128, dim=0), all_labels


### 修改点 ###
# 将 evaluate_f1_score 函数扩展为 evaluate_metrics
def evaluate_metrics(classifier, gallery_loader, query_loader, device,
                     k=5, vote_tau=0.5, weighted=True):
    """
    - 计算 F1, Precision, Recall, 和 AUC 分数.
    - 余弦相似度检索（输出已 L2 归一化）
    - 阈值投票（相对 k 或相似度和），减少“并集投票”造成的假阳性
    - 兼容 query_loader 在运行时跳过缺失嵌入的样本
    """
    print("\n--- Starting Metrics Evaluation ---")
    gallery_embeddings, gallery_labels = get_all_embeddings_and_labels(classifier, gallery_loader, device)
    query_embeddings, query_labels = get_all_embeddings_and_labels(classifier, query_loader, device)

    default_metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0}

    if gallery_embeddings.numel() == 0 or len(gallery_labels) == 0:
        print("[Eval] Gallery is empty; cannot evaluate.")
        return default_metrics
    if query_embeddings.numel() == 0 or len(query_labels) == 0:
        print("[Eval] Query set is empty after runtime skips; returning 0.0 for all metrics.")
        return default_metrics

    sims = torch.matmul(query_embeddings, gallery_embeddings.T)  # (Q, G)
    topk_sims, topk_idx = sims.topk(k, largest=True, dim=1)

    # 初始化用于存储每个样本分数的列表
    f1_scores, precision_scores, recall_scores, auc_scores = [], [], [], []

    for i in tqdm(range(len(query_labels)), desc="Calculating Metrics"):
        true_set = set(query_labels[i])
        
        # --- 1. 计算 Precision, Recall, F1 ---
        counter = defaultdict(float)
        weights_sum = 0.0
        for j, idx in enumerate(topk_idx[i].tolist()):
            w = float(topk_sims[i, j].item()) if weighted else 1.0
            weights_sum += w
            for ec in gallery_labels[idx]:
                counter[ec] += w

        thr_ref = weights_sum if weighted else k
        thr = vote_tau * thr_ref
        pred_set = {ec for ec, score in counter.items() if score >= thr}
        if not pred_set and counter:
            pred_set = {max(counter.items(), key=lambda x: x[1])[0]}

        tp = len(true_set & pred_set)
        prec = tp / len(pred_set) if pred_set else 0.0
        rec = tp / len(true_set) if true_set else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        f1_scores.append(f1)
        precision_scores.append(prec)
        recall_scores.append(rec)

        # --- 2. 计算 AUC ---
        # 对于多标签任务，我们为每个样本计算AUC，然后取平均
        # y_true: 对于所有在近邻中出现的EC号，它们是否是真实标签 (1/0)
        # y_score: 这些EC号从投票中获得的分数
        if counter:
            all_scored_ecs = list(counter.keys())
            y_true_auc = [1 if ec in true_set else 0 for ec in all_scored_ecs]
            y_score_auc = [counter[ec] for ec in all_scored_ecs]
            
            # AUC计算要求y_true中至少有两个类别（0和1）
            if len(set(y_true_auc)) > 1:
                try:
                    sample_auc = roc_auc_score(y_true_auc, y_score_auc)
                    auc_scores.append(sample_auc)
                except ValueError:
                    # 如果出现错误（虽然罕见），跳过此样本的AUC计算
                    pass

    # --- 3. 计算所有指标的宏平均值 ---
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    macro_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    macro_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.0 # 如果没有可计算的AUC样本，则为0

    print("--- Evaluation Finished ---")
    
    # 返回包含所有指标的字典
    return {
        'f1': macro_f1,
        'precision': macro_precision,
        'recall': macro_recall,
        'auc': macro_auc
    }
# ########### #

# -----------------------------
# 4. Helpers for Loading & Mapping
# -----------------------------
def load_and_create_label_map(csv_path):
    """
    更稳健的读取：自动识别分隔符，兼容 NaN，过滤空标签
    期望列：'Entry', 'EC number'（EC 多个时以 ';' 分隔）
    """
    if not os.path.exists(csv_path):
        return {}, []
    df = pd.read_csv(csv_path, sep=None, engine='python')
    required = {'Entry', 'EC number'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} 缺少必要列：{required - set(df.columns)}")

    df['EC number'] = df['EC number'].fillna('').astype(str)
    df['EC number'] = df['EC number'].apply(lambda s: [t.strip() for t in s.split(';') if t.strip()])
    df = df[df['EC number'].map(len) > 0]  # 过滤空标签行

    id_to_ecs = {row['Entry']: row['EC number'] for _, row in df.iterrows()}
    return id_to_ecs, list(id_to_ecs.keys())


def augment_map_with_mutants(id_to_ecs_master, embedding_dir, train_ids_orig_only=None):
    """
    方案 A：让“训练原始ID”的变体”继承原始 ID 的标签，仅用于训练池与正/负样本候选；
    评估时不强制纳入 gallery（保持保守），但外部测试集现在允许包含变体ID。
    """
    files = [f for f in os.listdir(embedding_dir) if f.endswith('.pt')]
    mutant_ids = [f[:-3] for f in files if '_' in f]
    add = {}
    train_orig_set = set(train_ids_orig_only) if train_ids_orig_only is not None else None

    for mid in mutant_ids:
        base = mid.split('_')[0]
        if base in id_to_ecs_master:
            if (train_orig_set is None) or (base in train_orig_set):
                add[mid] = id_to_ecs_master[base]

    if add:
        print(f"Augmenting label map with {len(add)} mutant IDs (train-origin only).")
        id_to_ecs_master.update(add)
    return id_to_ecs_master


def build_ec_to_ids(id_to_ecs_map):
    items = []
    for entry, ecs in id_to_ecs_map.items():
        for ec in ecs:
            items.append({'Entry': entry, 'EC number': ec})
    df = pd.DataFrame(items)
    if df.empty:
        return {}
    return df.groupby('EC number')['Entry'].apply(list).to_dict()


def filter_ids_with_embeddings(ids, embedding_dir, msg_prefix=""):
    ok, miss = [], []
    for sid in ids:
        if _has_embedding(embedding_dir, sid):
            ok.append(sid)
        else:
            miss.append(sid)
    if miss:
        print(f"{msg_prefix}Found {len(miss)} ids without embeddings; they are skipped.")
    return ok


# -----------------------------
# 5. Main
# -----------------------------
def main():
    config = {
        "csv_path": "./data/split100.csv",
        "dist_map_path": "./data/base/reports/distance_dict.pkl",
        "embedding_dir": "./data/base/protein_embeddings/",
        "batch_size": 64,
        "epochs": 50,
        "learning_rate": 3e-5,
        "device": torch.device('cuda:7' if torch.cuda.is_available() else 'cpu'),
        "model_save_path": 'best_classifier_model.pth',
        "num_workers": 4,
        "hard_negative_k": 5,
        "vote_k": 5,
        "vote_tau": 0.15,
        "vote_weighted": False,
        "grad_clip_norm": 1.0,
        "external_test_sets": {
            "price": "./data/price.csv",
            "new": "./data/new.csv"
        }
    }
    print(f"Using device: {config['device']}")

    # --- Step 1: Load maps (master) ---
    print("Loading and preparing data mappings (master)...")
    id_to_ecs_master, _ = load_and_create_label_map(config["csv_path"])
    for name, path in config["external_test_sets"].items():
        id_to_ecs_ext, _ = load_and_create_label_map(path)
        id_to_ecs_master.update(id_to_ecs_ext)

    # 载入 EC 距离
    with open(config["dist_map_path"], 'rb') as f:
        dist_matrix = pickle.load(f)

    # --- Step 2: Split by ORIGINAL IDs from main CSV ---
    _, all_main_ids = load_and_create_label_map(config["csv_path"])
    all_original_ids = [entry for entry in all_main_ids if '_' not in entry]
    random.shuffle(all_original_ids)

    n = len(all_original_ids)
    train_split_idx = int(0.8 * n)
    val_split_idx = int(0.9 * n)

    train_ids_orig = all_original_ids[:train_split_idx]
    val_ids_orig = all_original_ids[train_split_idx:val_split_idx]
    internal_test_ids_orig = all_original_ids[val_split_idx:]

    # 方案 A：将“训练原始ID”的变体加入 label map（只继承训练集的原始ID）
    id_to_ecs_master = augment_map_with_mutants(
        id_to_ecs_master,
        embedding_dir=config["embedding_dir"],
        train_ids_orig_only=train_ids_orig
    )

    # 用“扩充后的 master”重建 ec_to_ids（让变体也可作为正/负）
    ec_to_ids = build_ec_to_ids(id_to_ecs_master)

    # 训练池 = 训练原始ID + 对应变体（只取来自训练原始ID的变体）
    mutants_train = [sid for sid in id_to_ecs_master.keys()
                     if '_' in sid and sid.split('_')[0] in set(train_ids_orig)]
    train_pool_ids = train_ids_orig + mutants_train

    # 评估用的 gallery/val/test：保持原有安全策略
    train_ids_orig = filter_ids_with_embeddings(train_ids_orig, config["embedding_dir"],
                                                  msg_prefix="[Gallery] ")
    val_ids = filter_ids_with_embeddings(val_ids_orig, config["embedding_dir"],
                                           msg_prefix="[Val] ")
    internal_test_ids = filter_ids_with_embeddings(internal_test_ids_orig, config["embedding_dir"],
                                                     msg_prefix="[InternalTest] ")
    # 训练池也要过滤，避免 __getitem__ 报错
    train_pool_ids = filter_ids_with_embeddings(train_pool_ids, config["embedding_dir"],
                                                  msg_prefix="[TrainPool] ")

    print(f"\nTraining pool size (for triplet generation): {len(train_pool_ids)}")
    print(f"Validation set size: {len(val_ids)}")
    print(f"Internal Test set size: {len(internal_test_ids)}")

    # --- Step 3: Datasets & Loaders ---
    train_dataset = TripletDataset(
        train_pool_ids, id_to_ecs_master, ec_to_ids, dist_matrix,
        config["embedding_dir"], config["hard_negative_k"], missing_policy="skip"
    )
    gallery_dataset = EvaluationDataset(
        train_ids_orig, id_to_ecs_master, config["embedding_dir"], strict_embeddings=True
    )
    val_dataset = EvaluationDataset(
        val_ids, id_to_ecs_master, config["embedding_dir"], strict_embeddings=True
    )
    internal_test_dataset = EvaluationDataset(
        internal_test_ids, id_to_ecs_master, config["embedding_dir"], strict_embeddings=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True, persistent_workers=True,
        prefetch_factor=2
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True, persistent_workers=True,
        prefetch_factor=2, collate_fn=evaluation_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True, persistent_workers=True,
        prefetch_factor=2, collate_fn=evaluation_collate_fn
    )
    internal_test_loader = DataLoader(
        internal_test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True, persistent_workers=True,
        prefetch_factor=2, collate_fn=evaluation_collate_fn
    )

    # --- Step 4: Model / Loss / Optimizer ---
    classifier = EcClassifier(input_dim=1280).to(config['device'])
    cos = nn.CosineSimilarity(dim=-1)
    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - cos(x, y),
        margin=0.2
    )
    optimizer = optim.AdamW(classifier.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()
    best_val_f1 = -1.0

    # --- Step 5: Training & Eval Loop ---
    print("\nStarting model training...")
    for epoch in range(config['epochs']):
        classifier.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            anchor_1280 = batch['anchor'].to(config['device'], non_blocking=True)
            positive_1280 = batch['positive'].to(config['device'], non_blocking=True)
            negative_1280 = batch['negative'].to(config['device'], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                anchor_128 = classifier(anchor_1280)
                positive_128 = classifier(positive_1280)
                negative_128 = classifier(negative_1280)
                loss = criterion(anchor_128, positive_128, negative_128)

            scaler.scale(loss).backward()
            if config.get("grad_clip_norm") and config["grad_clip_norm"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(classifier.parameters(), config["grad_clip_norm"])

            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        epoch_loss = running_loss / max(1, len(train_loader))

        ### 修改点 ###
        # 调用新的评估函数并处理返回的字典
        val_metrics = evaluate_metrics(
            classifier, gallery_loader, val_loader, config['device'],
            k=config["vote_k"], vote_tau=config["vote_tau"], weighted=config["vote_weighted"]
        )
        val_f1 = val_metrics['f1']
        
        # 更新打印信息的格式
        print(f"Epoch {epoch+1}/{config['epochs']}, Training Loss: {epoch_loss:.4f}")
        print(f"Validation Metrics -> "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}")
        # ########### #

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(classifier.state_dict(), config['model_save_path'])
            print(f"Validation F1 improved! Model saved to {config['model_save_path']}")

        train_dataset.resample_triplets()

    # --- Step 6: Final Testing ---
    print("\nTraining complete. Loading best model for final testing...")
    classifier.load_state_dict(torch.load(config['model_save_path'], map_location=config['device']))
    classifier.to(config['device'])

    ### 修改点 ###
    # 更新内部测试集的评估和结果打印
    print("\n--- Evaluating on Internal Test Set ---")
    internal_test_metrics = evaluate_metrics(
        classifier, gallery_loader, internal_test_loader, config['device'],
        k=config["vote_k"], vote_tau=config["vote_tau"], weighted=config["vote_weighted"]
    )
    print("\n>> Final Internal Test Metrics:")
    for metric_name, score in internal_test_metrics.items():
        print(f"   - {metric_name.capitalize()}: {score:.4f}")
    # ########### #

    # --- External Test Sets: 不做预过滤 ---
    for name, path in config["external_test_sets"].items():
        print(f"\n--- Evaluating on External Test Set: {name.upper()} ---")
        id_to_ecs_ext, ext_test_ids = load_and_create_label_map(path)

        ext_test_dataset = EvaluationDataset(
            ext_test_ids, id_to_ecs_master, config["embedding_dir"], strict_embeddings=False
        )
        ext_test_loader = DataLoader(
            ext_test_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True, persistent_workers=True,
            prefetch_factor=2, collate_fn=evaluation_collate_fn
        )

        ### 修改点 ###
        # 更新外部测试集的评估和结果打印
        ext_test_metrics = evaluate_metrics(
            classifier, gallery_loader, ext_test_loader, config['device'],
            k=config["vote_k"], vote_tau=config["vote_tau"], weighted=config["vote_weighted"]
        )
        print(f"\n>> Final Metrics on {name.upper()} Test Set:")
        for metric_name, score in ext_test_metrics.items():
            print(f"   - {metric_name.capitalize()}: {score:.4f}")
        # ########### #


if __name__ == '__main__':
    main()