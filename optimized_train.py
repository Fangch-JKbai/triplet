import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from dataset import triplet, collate_fn
from esm import pretrained
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
import os
from functools import partial
from utils import update_map, CosineTripletLoss
from ec import ec
import time


def evaluate(model, dataloader, criterion, device, repr_layer):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            anchor_tokens = batch['anchor']['tokens'].to(device, dtype=torch.long)
            positive_tokens = batch['positive']['tokens'].to(device, dtype=torch.long)
            negative_tokens = batch['negative']['tokens'].to(device, dtype=torch.long)
            b = anchor_tokens.size(0)

            all_tokens = torch.cat([anchor_tokens, positive_tokens, negative_tokens], dim=0)
            all_embed = model(all_tokens, repr_layers=[repr_layer])['representations'][repr_layer][:, 1:-1, :].mean(1)
            
            anchor_embed = all_embed[:b]
            positive_embed = all_embed[b:2*b]
            negative_embed = all_embed[2*b:]

            loss = criterion(anchor_embed, positive_embed, negative_embed)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)


def main():
    # --- 超参数和配置 ---
    csv_file = '/home/fangchh/workdir/CLEAN/app/data/split10.csv'
    dist_map = 'distance_dict.pkl'
    batch_size = 16
    epochs = 1
    layer_count = 33
    learning_rate = 1e-6
    device = torch.device('cuda:7')
    best_val_loss = float('inf')
    model_save_path = 'ft_triplet_finetuned_model.pth'
    cpt_model_path = '/home/fangchh/workdir/cpt_models/n2/esm1b_t33_650M_UR50S_last_six_layers_and_lm_head_epoch0.pth'

    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")

    # --- 模型和 Alphabet ---
    print("Loading ESM model...")
    start_time = time.time()
    model, alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
    cpt_checkpoint = torch.load(cpt_model_path, map_location=device)
    cpt_state_dict = cpt_checkpoint.get('model_state_dict', cpt_checkpoint)
    model.load_state_dict(cpt_state_dict, strict=False)
    # 确保模型使用float32精度
    model = model.float().to(device)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
    print(f"Model loaded in {time.time() - start_time:.2f}s")

    # --- 数据准备 ---
    print("Creating dataset (this will be cached for subsequent runs)...")
    start_time = time.time()
    full_dataset = triplet(csv_file, dist_map, hard_negative_k=5, preload_sequences=True)
    print(f"Dataset created in {time.time() - start_time:.2f}s")
    
    data_size = len(full_dataset)
    train_size = int(0.8 * data_size)
    val_size = int(0.1 * data_size)

    torch.manual_seed(42)
    all_indices = list(range(data_size))
    random.shuffle(all_indices)
    train_idx = all_indices[: train_size]
    val_idx   = all_indices[train_size : train_size + val_size]
    test_idx  = all_indices[train_size + val_size :]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)
    test_dataset  = Subset(full_dataset, test_idx)
    
    # 使用 functools.partial 来包装 collate_fn
    collate_fn_with_converter = partial(collate_fn, batch_converter=batch_converter)

    # Create simple DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_with_converter
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_with_converter
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_with_converter
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 解冻最后几层用于微调 ---
    for name, param in model.named_parameters():
        param.requires_grad = False

    layers_to_unfreeze = [f"layers.{i}." for i in range(layer_count - 2, layer_count)]
    for name, param in model.named_parameters():
        if any(pat in name for pat in layers_to_unfreeze):
            param.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # --- 损失函数、优化器 ---
    criterion = CosineTripletLoss(margin=0.2, reduction='mean')
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )

    print("开始微调训练...")
    # --- 训练循环 ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        print("开始更新dist_map")
        update_start_time = time.time()
        if epoch == 0:
            update_map(model=model, checkpoint_path=None, device=device, batch_size=32)  # Even larger batch for embedding
        else:
            # Load checkpoint into existing model instead of creating new one
            checkpoint = torch.load(model_save_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
            update_map(model=model, checkpoint_path=None, device=device, batch_size=32)
        print(f"Distance map updated in {time.time() - update_start_time:.2f}s")
            
        # Simple training loop
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device and ensure correct dtype
            anchor_tokens = batch['anchor']['tokens'].to(device, dtype=torch.long)
            positive_tokens = batch['positive']['tokens'].to(device, dtype=torch.long)
            negative_tokens = batch['negative']['tokens'].to(device, dtype=torch.long)

            optimizer.zero_grad()

            # Forward pass
            all_tokens = torch.cat([anchor_tokens, positive_tokens, negative_tokens], dim=0)
            all_embed = model(all_tokens, repr_layers=[layer_count])['representations'][layer_count][:, 1:-1, :].mean(1)
            
            b = anchor_tokens.size(0)
            anchor_embed = all_embed[:b]
            positive_embed = all_embed[b:2*b]
            negative_embed = all_embed[2*b:3*b]
            
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * anchor_tokens.size(0)
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        
        val_start_time = time.time()
        val_loss = evaluate(model, val_loader, criterion, device, layer_count)
        val_time = time.time() - val_start_time
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs}, 训练损失: {epoch_loss:.4f}, 验证损失: {val_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s, Validation time: {val_time:.2f}s")

        # --- 保存最佳模型 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"验证损失降低，模型已保存至 {model_save_path}")
            
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("训练完成。")
    
    # --- 测试 ---
    print("开始在测试集上评估最佳模型...")
    test_start_time = time.time()
    checkpoint = torch.load(model_save_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    test_loss = evaluate(model, test_loader, criterion, device, layer_count)
    test_time = time.time() - test_start_time
    print(f"最终测试集损失: {test_loss:.4f}, 测试时间: {test_time:.2f}s")


if __name__ == '__main__':
    main()