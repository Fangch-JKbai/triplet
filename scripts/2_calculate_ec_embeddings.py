# -*- coding: utf-8 -*-
"""
Step 2: Calculate average embeddings for each unique EC number.
(No TQDM progress bar for clean logging)
"""
import torch
import pandas as pd
import os
import pickle
import argparse
# from tqdm import tqdm # TQDM has been removed
from collections import defaultdict

def generate_ec_average_embeddings(csv_path: str, embedding_dir: str, output_dir: str):
    """
    Calculates the average embedding for each unique EC number.
    """
    print("--- Starting EC Average Embedding Generation ---")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, sep='\t')
        df['EC number'] = df['EC number'].astype(str).str.split(';').apply(lambda x: [i.strip() for i in x])
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    ec_to_ids_map = defaultdict(list)
    df_exploded = df.explode('EC number')
    for _, row in df_exploded.iterrows():
        ec_to_ids_map[row['EC number']].append(row['Entry'])
    
    total_ecs = len(ec_to_ids_map)
    print(f"Found {total_ecs} unique EC numbers to process.")

    processed_count = 0
    # Replaced TQDM with a simple loop and counter
    for i, (ec_id, seq_ids) in enumerate(ec_to_ids_map.items()):
        ec_embeddings = []
        for seq_id in seq_ids:
            embedding_path = os.path.join(embedding_dir, f"{seq_id}.pt")
            if os.path.exists(embedding_path):
                try:
                    embedding_data = torch.load(embedding_path, map_location='cpu')
                    ec_embeddings.append(embedding_data['mean_representations'])
                except Exception as e:
                    print(f"\nWarning: Could not load or process embedding for {seq_id}: {e}")

        if ec_embeddings:
            average_embedding = torch.stack(ec_embeddings).mean(dim=0)
            output_data = {
                'ec_id': ec_id,
                'average_embedding': average_embedding,
                'source_ids': seq_ids
            }
            safe_ec_id = ec_id.replace('.', '_').replace('-', 'dash')
            output_path = os.path.join(output_dir, f"{safe_ec_id}.pkl")

            with open(output_path, 'wb') as f:
                pickle.dump(output_data, f)
            processed_count += 1
        
        # Optional: Print progress every N ECs
        if (i + 1) % 500 == 0:
            print(f"  ... processed {i + 1} / {total_ecs} EC numbers")

    print(f"\n[Step 2] Successfully generated and saved {processed_count} EC average embeddings in '{output_dir}'. ✨")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average embeddings for EC numbers.")
    
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the main CSV file (e.g., 'split100.csv').")
    parser.add_argument("--embedding-dir", type=str, required=True, help="Directory where the individual protein .pt embeddings are stored.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the final EC average embedding .pkl files.")
    
    args = parser.parse_args()
    
    generate_ec_average_embeddings(
        csv_path=args.csv_path,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir
    )
