import torch
import random
import numpy as np
from torch.utils.data import Sampler
from typing import List, Dict

class PKSampler(Sampler):
    """
    PK Sampler: A batch sampling strategy for contrastive learning.
    It ensures each batch contains P distinct classes with K samples each.
    """
    def __init__(self, ec_to_indices: Dict[str, List[int]], p: int, k: int):
        """
        Args:
            ec_to_indices: A dictionary mapping EC number to a list of sample indices.
            p (int): The number of distinct classes (persons) per batch.
            k (int): The number of samples (kindred) per class.
        """
        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"p must be a positive integer, but got {p}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, but got {k}")

        self.ec_to_indices = ec_to_indices
        self.p = p
        self.k = k

        # Filter out classes with fewer than K samples, as they can't be sampled without replacement.
        # For simplicity, we can require at least K samples. A more advanced version might handle this differently.
        self.ecs_with_enough_samples = [
            ec for ec, indices in self.ec_to_indices.items() if len(indices) >= self.k
        ]
        
        if not self.ecs_with_enough_samples:
            raise ValueError("No classes have enough samples (>= k) to form a batch.")
            
        self.num_classes = len(self.ecs_with_enough_samples)
        self.num_batches = self.num_classes // self.p

    def __iter__(self):
        # Shuffle the classes at the beginning of each epoch
        random.shuffle(self.ecs_with_enough_samples)
        
        # Iterate through the shuffled classes, creating batches
        for i in range(self.num_batches):
            batch_indices = []
            # Select P classes for the current batch
            class_batch = self.ecs_with_enough_samples[i * self.p : (i + 1) * self.p]
            
            for ec in class_batch:
                # Get all possible indices for the current class
                possible_indices = self.ec_to_indices[ec]
                
                # Randomly sample K indices without replacement
                sampled_indices = random.sample(possible_indices, self.k)
                batch_indices.extend(sampled_indices)
            
            yield batch_indices

    def __len__(self):
        return self.num_batches