# 文件名: scripts/1_generate_protein_embeddings.py
# 优化版本：使用 Masking 策略进行数据增强 + 统一加载模型

import argparse
import torch
import os
import csv
import math
import random
import sys
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict, Set, Tuple

# 添加EnzSub模型路径
sys.path.append("/home/fangchh/workdir/enzyme/sub")
from model import EnzSubModelForDownstream


# ============== 模型配置（统一管理）==============

ESM_MODELS_CONFIG = {
    # ESM-2 原始预训练模型
    "ESM_2": {
        "checkpoint_path": None,
        "load_lora": False,
    },
    # CPT only (Stage-1 continued pretraining)
    "ESM_CPT": {
        "checkpoint_path": "/home/fangchh/workdir/enzyme/cpt_span/output/span_cpt_2gpu_15mask_6layer_final/checkpoint_epoch_2.pt",
        "load_lora": False,
    },
    # ESM-2 + LoRA (Stage-2 only, 无CPT基础)
    "ESM_SUB": {
        "checkpoint_path": "/home/fangchh/workdir/enzyme/sub/outputs/enzsub_esm2/checkpoint_epoch10.pt",
        "load_lora": True,
    },
    # CPT + LoRA (Stage-1 + Stage-2)
    "ESM_CPT_SUB": {
        "checkpoint_path": "/home/fangchh/workdir/enzyme/sub/outputs/enzsub_cpt/checkpoint_epoch10.pt",
        "load_lora": True,
    },
}

# ============== 数据增强功能区 (修改为 Masking 策略) ==============

def get_ec_id_dict(csv_path: str) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """解析CSV文件，创建序列ID和EC号之间的映射。"""
    id_ec_map: Dict[str, List[str]] = {}
    ec_id_map: Dict[str, Set[str]] = {}
    if not csv_path or not os.path.exists(csv_path):
        print("Warning: CSV path not provided or file not found. Skipping mutation logic.")
        return {}, {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) < 2:
                continue
            seq_id, ec_numbers_str = row[0], row[1]
            ec_numbers = [ec.strip() for ec in ec_numbers_str.split(';')]
            id_ec_map[seq_id] = ec_numbers
            for ec in ec_numbers:
                if ec not in ec_id_map:
                    ec_id_map[ec] = set()
                ec_id_map[ec].add(seq_id)
    return id_ec_map, ec_id_map


def apply_random_masking(seq: str) -> str:
    """
    对序列进行随机 Masking (方法一)。
    策略：
    1. 基于正态分布(mu=0.1, sigma=0.02)确定 mask 比例。
    2. 随机选择位置替换为 '<mask>'。
    """
    seq_len = len(seq)
    seq_list = list(seq) # 转为列表以便修改
    
    # 1. 确定突变率 (使用 numpy 生成正态分布随机数)
    mu, sigma = 0.10, 0.02
    mut_rate = np.random.normal(mu, sigma)
    
    # 限制突变率在合理范围内 (例如 0 到 100%)
    mut_rate = max(0.0, min(1.0, mut_rate))
    
    # 2. 计算需要 Mask 的数量
    num_masks = math.ceil(seq_len * mut_rate)
    
    # 3. 随机选择位置 (使用 sample 无放回采样，防止重复 mask 同一个位置)
    if num_masks > 0:
        positions = random.sample(range(seq_len), k=min(num_masks, seq_len))
        for pos in positions:
            seq_list[pos] = "X" # 插入 ESM 专用的 mask token
            
    return "".join(seq_list)


def create_mutated_fasta(
    csv_path: str, 
    output_fasta_path: str,
    num_augmentations: int = 10
) -> bool:
    """为那些只对应一个序列的EC号生成 Masked 序列，作为数据增强。"""
    print("\n--- Starting Data Augmentation: Masking Sequences ---")
    id_ec_map, ec_id_map = get_ec_id_dict(csv_path)
    if not ec_id_map:
        return False

    # 找出只有一条序列的 EC 号对应的 Seq ID
    single_ec_ids = {ids.pop() for ec, ids in ec_id_map.items() if len(ids) == 1}
    print(f"Found {len(single_ec_ids)} sequences belonging to single-entry ECs to augment.")
    if not single_ec_ids:
        return False

    try:
        with open(csv_path, 'r', encoding='utf-8') as csv_file, \
             open(output_fasta_path, 'w', encoding='utf-8') as fasta_file:
            
            reader = csv.reader(csv_file, delimiter='\t')
            header = next(reader)
            # 自动寻找 Sequence 列的索引，如果找不到默认第3列(索引2)
            try:
                seq_col_idx = header.index('Sequence')
            except ValueError:
                seq_col_idx = 2 

            total_generated = 0

            for row in reader:
                seq_id = row[0]
                if seq_id in single_ec_ids:
                    sequence = row[seq_col_idx].strip()
                    
                    # 对每一条需要增强的序列，生成 num_augmentations 个变体
                    for i in range(num_augmentations):
                        masked_seq = apply_random_masking(sequence)
                        
                        # 写入 FASTA
                        fasta_file.write(f">{seq_id}_mask_{i}\n")
                        fasta_file.write(f"{masked_seq}\n")
                        total_generated += 1
            
        print(f"Successfully generated {total_generated} masked sequences in: {output_fasta_path}")
        return True
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Error during masked FASTA creation: {e}")
        return False


# ============== 模型加载与嵌入生成区 ==============

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model_by_name(model_name: str, device: str = "cuda:0"):
    if model_name not in ESM_MODELS_CONFIG:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(ESM_MODELS_CONFIG.keys())}")
    
    config = ESM_MODELS_CONFIG[model_name]
    
    print(f"\n--- Loading Model: {model_name} ---")
    print(f"    checkpoint_path: {config['checkpoint_path']}")
    print(f"    load_lora: {config['load_lora']}")
    
    model = EnzSubModelForDownstream(
        checkpoint_path=config['checkpoint_path'],
        load_lora=config['load_lora'],
        device=device
    )
    
    print(f"--- ✅ Model '{model_name}' loaded successfully! ---")
    return model


def read_fasta(fasta_file: str) -> Tuple[List[str], List[str]]:
    """读取FASTA文件，返回序列ID和序列列表"""
    sequences, seq_ids = [], []
    with open(fasta_file) as f:
        current_id, current_seq = None, []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id:
                    sequences.append("".join(current_seq))
                    seq_ids.append(current_id)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences.append("".join(current_seq))
            seq_ids.append(current_id)
    return seq_ids, sequences


def generate_embeddings(
    model: EnzSubModelForDownstream,
    fasta_file: str,
    output_dir: str,
    batch_size: int = 16,
    max_len: int = 1022
):
    ensure_dirs(output_dir)
    
    seq_ids, sequences = read_fasta(fasta_file)
    print(f"Found {len(sequences)} sequences in {fasta_file}.")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Embedding {os.path.basename(fasta_file)}"):
            batch_ids = seq_ids[i:i + batch_size]
            batch_seqs = [s[:max_len] if len(s) > max_len else s for s in sequences[i:i + batch_size]]
            
            data = list(zip(batch_ids, batch_seqs))
            embeddings = model.get_embedding(data)
            
            for j, seq_id in enumerate(batch_ids):
                out_path = os.path.join(output_dir, f"{seq_id}.pt")
                torch.save({
                    "label": seq_id, 
                    "mean_representations": embeddings[j].cpu()
                }, out_path)
    
    print(f"All embeddings for {os.path.basename(fasta_file)} saved in: {output_dir}")


# ============== 主函数区 ==============

def main():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using unified EnzSubModelForDownstream with Masking Augmentation."
    )
    
    # 模型参数
    parser.add_argument("--model-name", type=str, required=True,
                        choices=list(ESM_MODELS_CONFIG.keys()),
                        help="Model name to use (ESM_2, ESM_CPT, ESM_SUB, ESM_CPT_SUB)")
    
    # 数据参数
    parser.add_argument("--fasta-files", type=str, nargs='+', required=True,
                        help="List of input FASTA files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save embeddings.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference.")
    parser.add_argument("--device", type=str, 
                        default='cuda:9' if torch.cuda.is_available() else 'cpu',
                        help="Device to use.")

    # 数据增强参数
    parser.add_argument("--run-mutation", action="store_true",
                        help="Set to run the sequence Masking data augmentation.")
    parser.add_argument("--csv-file", type=str,
                        help="[For Augmentation] CSV file containing sequences and EC numbers.")
    
    args = parser.parse_args()
    
    # --- 数据增强步骤 (Masking) ---
    if args.run_mutation:
        if not args.csv_file:
            raise ValueError("--csv-file is required when using --run-mutation.")
        
        # 修改输出文件名为 masked_sequences.fasta 以示区分
        mutated_fasta_path = os.path.join(os.path.dirname(args.csv_file), "masked_sequences.fasta")
        success = create_mutated_fasta(args.csv_file, mutated_fasta_path)
        
        if success and mutated_fasta_path not in args.fasta_files:
            args.fasta_files.append(mutated_fasta_path)

    # --- 模型加载 ---
    model = load_model_by_name(args.model_name, args.device)

    # --- Embedding 生成 ---
    print("\n--- Starting Main Embedding Generation Task ---")
    for fasta in args.fasta_files:
        if not os.path.exists(fasta):
            print(f"Warning: FASTA file not found, skipping: {fasta}")
            continue
        generate_embeddings(
            model=model,
            fasta_file=fasta,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )

    print("\n✅ All tasks completed successfully!")


if __name__ == "__main__":
    main()