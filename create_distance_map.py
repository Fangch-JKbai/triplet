import os
import pickle
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. Corrected Data Loading Function ---
def load_all_ec_embeddings(directory="ec"):
    """
    Loads all pre-computed EC average embeddings from a directory.

    This function is designed to load the simple dictionary format created by the
    corrected `ec_pkl_generator.py` script, making it robust and efficient.
    """
    print(f"--- Loading all EC average embeddings from '{directory}' ---")
    
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return {}

    # Find all relevant .pkl files
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"Warning: No .pkl files found in '{directory}'.")
        return {}
        
    embeddings_dict = {}
    
    for filename in tqdm(pkl_files, desc="Loading EC embeddings"):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'rb') as f:
                # Load the simple dictionary
                data = pickle.load(f)
            
            # Extract the required information
            ec_id = data.get('ec_id')
            avg_embedding = data.get('average_embedding')
            
            # Add to our dictionary if the data is valid
            if ec_id and avg_embedding is not None:
                embeddings_dict[ec_id] = avg_embedding.cpu().detach()
            else:
                print(f"\nWarning: Skipping {filename} due to missing 'ec_id' or 'average_embedding'.")

        except Exception as e:
            print(f"\nWarning: Failed to load or process file {filename}: {e}")
            continue
            
    print(f"--- Loading complete! Found {len(embeddings_dict)} valid EC average embeddings. ---")
    return embeddings_dict

# --- 2. Distance Matrix Calculation (Largely Unchanged, Logic is Good) ---
def calculate_distance_matrix(embeddings_dict, use_gpu=True):
    """
    Calculates the pairwise cosine distance between all embedding vectors.
    Uses vectorized computation and GPU acceleration for performance.
    """
    # Sort EC IDs to ensure a consistent matrix order
    ec_ids = sorted(embeddings_dict.keys())
    num_embeddings = len(ec_ids)
    
    if num_embeddings < 2:
        return None

    print(f"\n--- Calculating distance matrix for {num_embeddings}x{num_embeddings} embeddings... ---")
    
    # Stack all embeddings into a single tensor [num_embeddings, embedding_dim]
    embeddings_list = [embeddings_dict[ec_id] for ec_id in ec_ids]
    embeddings_matrix = torch.stack(embeddings_list)
    
    # Move to GPU if available
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    embeddings_matrix = embeddings_matrix.to(device)
    print(f"Calculating on device: {device}")
    
    with torch.no_grad():
        # L2 normalize the embeddings for cosine similarity calculation
        embeddings_normalized = torch.nn.functional.normalize(embeddings_matrix, p=2, dim=1)
        
        # Calculate cosine similarity matrix: A @ A.T
        similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
        
        # Cosine distance = 1 - Cosine similarity
        distance_matrix_tensor = 1 - similarity_matrix
        
        # Clamp values to avoid floating point inaccuracies (e.g., -1e-7)
        distance_matrix_tensor = torch.clamp(distance_matrix_tensor, min=0.0)
        
        # Ensure the diagonal is exactly zero
        distance_matrix_tensor.fill_diagonal_(0.0)
    
    # Convert to a pandas DataFrame for easy use
    distance_matrix = pd.DataFrame(
        distance_matrix_tensor.cpu().numpy(),
        index=ec_ids,
        columns=ec_ids
    )
    
    return distance_matrix

# --- 3. Heatmap Plotting (Largely Unchanged, Logic is Good) ---
def plot_heatmap(distance_matrix, output_filename="distance_heatmap.png"):
    """
    Visualizes the distance matrix as a heatmap using Seaborn and saves it.
    """
    print("\n--- Generating heatmap... ---")
    
    plt.figure(figsize=(12, 10))
    
    # For large matrices, don't show tick labels to keep it clean
    show_labels = len(distance_matrix) <= 50
    
    sns.heatmap(
        distance_matrix, 
        cmap='viridis_r', # '_r' reverses the colormap (lower is darker)
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

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    EC_EMBEDDING_DIR = "./ec/"
    OUTPUT_DICT_PATH = "./data/distance_dict.pkl"
    
    # Step 1: Load all EC average embeddings
    embeddings = load_all_ec_embeddings(directory=EC_EMBEDDING_DIR)
    
    if len(embeddings) > 1:
        # Step 2: Calculate the distance matrix
        dist_matrix = calculate_distance_matrix(embeddings)
        
        # Step 3: Save the distance matrix as a dictionary for the training script
        # The training script needs a dictionary format: {ec_id: {other_ec_id: distance, ...}}
        os.makedirs(os.path.dirname(OUTPUT_DICT_PATH), exist_ok=True)
        distance_dict = dist_matrix.to_dict('index')
        with open(OUTPUT_DICT_PATH, 'wb') as f:
            pickle.dump(distance_dict, f)
        print(f"\n--- Distance dictionary successfully saved to '{OUTPUT_DICT_PATH}' ---")
        
        # Step 4: (Optional) Visualize the matrix as a heatmap
        plot_heatmap(dist_matrix)
    else:
        print("\nError: Need at least two valid EC embeddings to generate a distance matrix.")
