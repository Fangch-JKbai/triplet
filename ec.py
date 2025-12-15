import torch
import pandas as pd
import os
import pickle
from tqdm import tqdm
from collections import defaultdict

def generate_ec_average_embeddings(csv_path: str, embedding_dir: str, output_dir: str):
    """
    Calculates the average embedding for each unique EC number using pre-computed
    individual protein embeddings.

    This script should be run AFTER `generate.py` has created all the individual
    protein embedding files (.pt).

    Args:
        csv_path (str): Path to the main CSV file (e.g., 'split10.csv').
        embedding_dir (str): Directory where the individual .pt embeddings are stored.
        output_dir (str): Directory to save the final ec average embedding .pkl files.
    """
    print("--- Starting EC Average Embedding Generation ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Load Data and Create Mappings (ONCE) ---
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, sep='\t')
        # Handle multi-label EC numbers by splitting the string
        df['EC number'] = df['EC number'].astype(str).str.split(';').apply(lambda x: [i.strip() for i in x])
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Create a mapping from EC number to a list of sequence IDs that have that EC number.
    # This is the most efficient way to group the data.
    ec_to_ids_map = defaultdict(list)
    df_exploded = df.explode('EC number')
    for _, row in df_exploded.iterrows():
        ec_to_ids_map[row['EC number']].append(row['Entry'])
    
    print(f"Found {len(ec_to_ids_map)} unique EC numbers to process.")

    # --- Step 2: Iterate Through Each EC, Load Embeddings, and Calculate Average ---
    processed_count = 0
    for ec_id, seq_ids in tqdm(ec_to_ids_map.items(), desc="Processing EC numbers"):
        
        ec_embeddings = []
        # For each sequence associated with the current EC number...
        for seq_id in seq_ids:
            embedding_path = os.path.join(embedding_dir, f"{seq_id}.pt")
            
            # ...load its pre-computed embedding.
            if os.path.exists(embedding_path):
                try:
                    embedding_data = torch.load(embedding_path, map_location='cpu')
                    ec_embeddings.append(embedding_data['mean_representations'])
                except Exception as e:
                    print(f"\nWarning: Could not load or process embedding for {seq_id}: {e}")
            # else:
            #     print(f"\nWarning: Embedding file not found for {seq_id}. Skipping.")

        # --- Step 3: Calculate the Average and Save ---
        if ec_embeddings:
            # Stack all embeddings for this EC into a single tensor and calculate the mean
            average_embedding = torch.stack(ec_embeddings).mean(dim=0)
            
            # Prepare data for saving in a simple, robust format
            output_data = {
                'ec_id': ec_id,
                'average_embedding': average_embedding,
                'source_ids': seq_ids # Optionally store which IDs were used
            }
            
            # Create a safe filename
            safe_ec_id = ec_id.replace('.', '_').replace('-', 'dash')
            output_path = os.path.join(output_dir, f"{safe_ec_id}.pkl")

            with open(output_path, 'wb') as f:
                pickle.dump(output_data, f)
            
            processed_count += 1
        # else:
        #     print(f"\nSkipping {ec_id} as no valid embeddings were found for its sequences.")

    print(f"\n--- Processing Complete ---")
    print(f"Successfully generated and saved {processed_count} EC average embedding files in '{output_dir}'. ✨")


if __name__ == "__main__":
    # --- Configuration ---
    # All paths should be relative to your project's root directory.
    CSV_FILE_PATH = "./data/split100.csv"
    EMBEDDING_INPUT_DIR = "./data/esm2_base_embeddings" # Directory with individual .pt files
    EC_PKL_OUTPUT_DIR = "./ec/" # Directory to save the final ec_*.pkl files

    generate_ec_average_embeddings(
        csv_path=CSV_FILE_PATH,
        embedding_dir=EMBEDDING_INPUT_DIR,
        output_dir=EC_PKL_OUTPUT_DIR
    )
