import torch
from torch.utils.data import Dataset
import os
import random
from tqdm import tqdm
from collections import defaultdict

# --- 1. Dataset for EVALUATION (Validation & Test Sets) ---
# Its job is simple: return one protein embedding and its true labels.
class EvaluationDataset(Dataset):
    """
    A dataset for evaluation purposes (validation, testing, and gallery building).
    It takes a list of protein IDs for a specific data split.
    Each item is a dictionary containing the pre-computed 1280-dim embedding and
    its corresponding list of multi-label EC numbers.
    """
    def __init__(self, ids, id_to_ecs_map, embedding_dir):
        self.ids = ids
        self.id_to_ecs_map = id_to_ecs_map
        self.embedding_dir = embedding_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        
        # The true labels belong to the original, non-mutated protein
        original_id = seq_id.split('_')[0]
        labels = self.id_to_ecs_map[original_id]
        
        # Load the pre-computed 1280-dim embedding from the .pt file
        embedding_path = os.path.join(self.embedding_dir, f"{seq_id}.pt")
        embedding_data = torch.load(embedding_path, map_location='cpu')
        embedding = embedding_data['mean_representations']

        return {"embedding_1280": embedding, "labels": labels}


# --- 2. Dataset for TRAINING ---
# Its job is to generate and return (anchor, positive, negative) triplets.
class TripletDataset(Dataset):
    """
    A dataset for training with Triplet Loss.
    It takes the pool of training IDs and the pre-computed distance dictionary
    to perform hard negative mining on-the-fly.
    Each item is a dictionary containing the anchor, positive, and negative
    pre-computed 1280-dim embeddings.
    """
    def __init__(self, train_ids, id_to_ecs_map, ec_to_ids_map, dist_matrix, embedding_dir, hard_negative_k=5):
        super().__init__()
        # It now receives the specific pool of IDs to use for triplet generation
        self.train_ids = train_ids
        self.id_to_ecs_map = id_to_ecs_map
        self.ec_to_ids_map = ec_to_ids_map
        self.dist_matrix = dist_matrix
        self.embedding_dir = embedding_dir
        self.hard_negative_k = hard_negative_k
        
        # The triplet generation is now a clean, internal method
        self.triplets = self._generate_triplets()

    def _generate_triplets(self):
        """
        Generates triplets ONLY from the provided pool of training IDs to prevent data leakage.
        """
        print("Generating training triplets using hard negative mining...")
        triplets = []
        
        # Pre-compute hard negative ECs for faster lookup
        ec_hard_negatives = {}
        for ec in self.ec_to_ids_map.keys():
            neighbor_ecs = self.dist_matrix.get(ec, {})
            # Sort neighbors by distance (ascending)
            distances = sorted(neighbor_ecs.items(), key=lambda x: x[1])
            ec_hard_negatives[ec] = [d[0] for d in distances[:self.hard_negative_k]]

        # Use a set for faster ID lookups
        train_ids_set = set(self.train_ids)

        # Iterate ONLY over the training IDs to select anchors
        for a_id in tqdm(self.train_ids, desc="Generating Triplets"):
            anchor_ec_list = self.id_to_ecs_map.get(a_id, [])
            for chosen_anchor_ec in anchor_ec_list:
                # Find positives from the same EC class, also within the training set
                positive_candidates = [
                    pid for pid in self.ec_to_ids_map.get(chosen_anchor_ec, []) 
                    if pid != a_id and pid in train_ids_set
                ]
                if not positive_candidates:
                    continue
                
                # Find hard negatives using the distance dictionary
                hard_negative_ecs = ec_hard_negatives.get(chosen_anchor_ec, [])
                negative_candidates = []
                for neg_ec in hard_negative_ecs:
                    candidates = [
                        nid for nid in self.ec_to_ids_map.get(neg_ec, []) 
                        if nid in train_ids_set
                    ]
                    negative_candidates.extend(candidates)
                
                if positive_candidates and negative_candidates:
                    p_id = random.choice(positive_candidates)
                    neg_id = random.choice(negative_candidates)
                    triplets.append((a_id, p_id, neg_id))

        print(f"Generated {len(triplets)} triplets for training.")
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_id, p_id, n_id = self.triplets[idx]
        
        # Load the pre-computed 1280-dim embeddings for the triplet
        a_embed = torch.load(os.path.join(self.embedding_dir, f"{a_id}.pt"), map_location='cpu')['mean_representations']
        p_embed = torch.load(os.path.join(self.embedding_dir, f"{p_id}.pt"), map_location='cpu')['mean_representations']
        n_embed = torch.load(os.path.join(self.embedding_dir, f"{n_id}.pt"), map_location='cpu')['mean_representations']
        
        return {"anchor": a_embed, "positive": p_embed, "negative": n_embed}

