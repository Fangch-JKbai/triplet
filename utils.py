import pickle
import torch
from esm import pretrained
import os
import glob
from ec import ec
import create_distance_map
import torch.nn.functional as F
import torch.nn as nn


pkl_path = 'ec'

def update_map(model=None, checkpoint_path=None, device=None, batch_size=16):
    # Reuse existing model instead of loading from scratch
    if model is None:
        model, alphabet = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
        batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
        
        # Load model checkpoint
        if checkpoint_path:
            print('加载上一轮模型更新')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
        else:
            print('加载原始模型更新')
        
        model.to(device)
        model.eval()
        model.float()
    else:
        # Reuse existing model and get alphabet
        alphabet = model.alphabet if hasattr(model, 'alphabet') else pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")[1]
        batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
        model.eval()
        model.float()

    pkl_files = glob.glob(os.path.join(pkl_path, '*.pkl'))
    
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            pkl = pickle.load(f)
        
        # Pre-allocate embeddings dict for better memory efficiency
        if not hasattr(pkl, 'embeddings') or pkl.embeddings is None:
            pkl.embeddings = {}
            
        with torch.no_grad():
            # Process in larger batches for better GPU utilization
            for i in range(0, len(pkl.idx), batch_size):
                batch_ids = pkl.idx[i:i+batch_size]
                batch_data = [(entry_id, pkl.sequences[entry_id]) for entry_id in batch_ids]

                _, _, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device, dtype=torch.long)

                # Simple forward pass without mixed precision
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                reps = results["representations"][33]
                
                # Vectorized mean calculation - much faster
                reps_mean = reps[:, 1:-1].mean(dim=1)  # Shape: [batch_size, hidden_dim]
                
                # Batch assignment to embeddings dict
                for j, entry_id in enumerate(batch_ids):
                    if j < len(reps_mean):
                        pkl.embeddings[entry_id] = reps_mean[j]

        if pkl.embeddings:
            all_embeddings_tensor = torch.stack(list(pkl.embeddings.values()))
            pkl.main_embedding = all_embeddings_tensor.mean(dim=0)
        else:
            pkl.main_embedding = None

        # Save updated pkl file
        with open(pkl_file, 'wb') as f:
            pickle.dump(pkl, f)
    
    embeddings_dict = create_distance_map.load_all_embeddings(directory='ec')

    if len(embeddings_dict) < 2:
        print("数据不足，无法进行距离计算。")
        return
    
    dist_matrix = create_distance_map.calculate_distance_matrix(embeddings_dict)
    print("\n--- 更新后的距离矩阵 (前5x5): ---")
    print(dist_matrix.iloc[:5, :5])

    pkl_filename = "distance_dict.pkl"
    distance_dict = dist_matrix.to_dict('index')
    with open(pkl_filename, 'wb') as f:
        pickle.dump(distance_dict, f)
    print(f"\n--- 距离字典数据已成功更新并保存为 '{pkl_filename}' ---")



class CosineTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity(dim=1)
        self.reduction = reduction

    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        pos_sim = self.cosine(anchor, positive)
        neg_sim = self.cosine(anchor, negative)
        loss = F.relu(self.margin + neg_sim - pos_sim)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
