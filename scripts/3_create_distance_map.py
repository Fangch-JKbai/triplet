# -*- coding: utf-8 -*-
"""
Step 3: Create a distance map from EC average embeddings.
(No TQDM progress bar for clean logging)
"""
import os
import pickle
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
# from tqdm import tqdm # TQDM has been removed

# --- (Data Loading, Calculation, and Plotting Functions) ---
def load_all_ec_embeddings(directory="ec"):
    """Loads all pre-computed EC average embeddings from a directory."""
    print(f"--- Loading all EC average embeddings from '{directory}' ---")
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return {}
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"Warning: No .pkl files found in '{directory}'.")
        return {}
    embeddings_dict = {}
    
    # Replaced TQDM with a simple loop
    for filename in pkl_files:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            ec_id = data.get('ec_id')
            avg_embedding = data.get('average_embedding')
            if ec_id and avg_embedding is not None:
                embeddings_dict[ec_id] = avg_embedding.cpu().detach()
            else:
                print(f"\nWarning: Skipping {filename} due to missing data.")
        except Exception as e:
            print(f"\nWarning: Failed to load or process file {filename}: {e}")
            continue
            
    print(f"--- Loading complete! Found {len(embeddings_dict)} valid EC average embeddings. ---")
    return embeddings_dict

def calculate_distance_matrix(embeddings_dict, use_gpu=True):
    """Calculates the pairwise cosine distance between all embedding vectors."""
    ec_ids = sorted(embeddings_dict.keys())
    num_embeddings = len(ec_ids)
    if num_embeddings < 2:
        return None
    print(f"\n--- Calculating distance matrix for {num_embeddings}x{num_embeddings} embeddings... ---")
    embeddings_list = [embeddings_dict[ec_id] for ec_id in ec_ids]
    embeddings_matrix = torch.stack(embeddings_list)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    embeddings_matrix = embeddings_matrix.to(device)
    print(f"Calculating on device: {device}")
    with torch.no_grad():
        embeddings_normalized = torch.nn.functional.normalize(embeddings_matrix, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
        distance_matrix_tensor = 1 - similarity_matrix
        distance_matrix_tensor = torch.clamp(distance_matrix_tensor, min=0.0)
        distance_matrix_tensor.fill_diagonal_(0.0)
    distance_matrix = pd.DataFrame(
        distance_matrix_tensor.cpu().numpy(),
        index=ec_ids,
        columns=ec_ids
    )
    return distance_matrix

def plot_heatmap(distance_matrix, output_filename="distance_heatmap.png"):
    """Visualizes the distance matrix as a heatmap and saves it."""
    print("\n--- Generating heatmap... ---")
    plt.figure(figsize=(12, 10))
    show_labels = len(distance_matrix) <= 50
    sns.heatmap(
        distance_matrix, 
        cmap='viridis_r', 
        xticklabels=show_labels,
        yticklabels=show_labels,
    )
    plt.title('Heatmap of Cosine Distance Between EC Average Embeddings', fontsize=16)
    if show_labels:
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"--- Heatmap successfully saved to '{output_filename}' ---")
    plt.close()

def main(args):
    """Main function to run the distance map creation."""
    embeddings = load_all_ec_embeddings(directory=args.ec_embedding_dir)
    
    if len(embeddings) < 2:
        print("\nError: Need at least two valid EC embeddings to generate a distance matrix.")
        return

    dist_matrix = calculate_distance_matrix(embeddings)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_dict_path = os.path.join(args.output_dir, "distance_dict.pkl")
    output_heatmap_path = os.path.join(args.output_dir, "distance_heatmap.png")
    
    distance_dict = dist_matrix.to_dict('index')
    with open(output_dict_path, 'wb') as f:
        pickle.dump(distance_dict, f)
    print(f"\n--- Distance dictionary successfully saved to '{output_dict_path}' ---")
    
    if not args.no_plot:
        plot_heatmap(dist_matrix, output_heatmap_path)
    
    print("\n[Step 3] Distance map creation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a distance map from EC average embeddings.")
    
    parser.add_argument("--ec-embedding-dir", type=str, required=True, help="Directory where the EC average .pkl files are stored.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the final distance dictionary and heatmap.")
    parser.add_argument("--no-plot", action='store_true', help="If set, do not generate the heatmap plot.")
    
    args = parser.parse_args()
    main(args)
