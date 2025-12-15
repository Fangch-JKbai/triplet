import torch
import esm
import os
import csv
import math
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List, Set
from triplet.scripts.mtl_esm import MTLModel
from peft import get_peft_model, LoraConfig

def get_ec_id_dict(csv_path: str) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """Parses a CSV file to create mappings between sequence IDs and EC numbers."""
    id_ec_map: Dict[str, List[str]] = {}
    ec_id_map: Dict[str, Set[str]] = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # Skip header
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
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return {}, {}
        
    return id_ec_map, ec_id_map

def mutate_sequence(seq: str, mutation_rate: float) -> str:
    """Applies random mutations to a sequence by replacing amino acids with '<mask>'."""
    seq_len = len(seq)
    num_mutations = min(math.ceil(seq_len * mutation_rate), seq_len)
    mutated_seq = list(seq)
    positions = random.sample(range(seq_len), k=num_mutations)
    for pos in positions:
        mutated_seq[pos] = '<mask>'
    return "".join(mutated_seq)

def create_mutated_fasta(
    csv_path: str, 
    output_fasta_path: str, 
    num_mutations_per_seq: int = 10,
    embedding_dir: str = './data/esm_data/'
) -> None:
    """
    Identifies sequences from single-entry ECs, mutates them, and saves to a new FASTA file.
    This is a data augmentation strategy.
    """
    print("Starting data augmentation: mutating single-sequence ECs...")
    id_ec_map, ec_id_map = get_ec_id_dict(csv_path)
    if not ec_id_map:
        return

    single_ec_set = {ec for ec, ids in ec_id_map.items() if len(ids) == 1}
    ids_to_mutate = set()
    for seq_id, ec_list in id_ec_map.items():
        if any(ec in single_ec_set for ec in ec_list):
            if not os.path.exists(os.path.join(embedding_dir, f"{seq_id}_0.pt")):
                ids_to_mutate.add(seq_id)

    print(f"Found {len(single_ec_set)} EC numbers with only one sequence.")
    print(f"Found {len(ids_to_mutate)} sequences to mutate.")

    if not ids_to_mutate:
        print("No new sequences to mutate. Skipping FASTA generation.")
        return

    try:
        with open(csv_path, 'r', encoding='utf-8') as csv_file, \
             open(output_fasta_path, 'w', encoding='utf-8') as fasta_file:
            
            reader = csv.reader(csv_file, delimiter='\t')
            header = next(reader) 
            seq_col_idx = header.index('Sequence')

            for row in reader:
                seq_id, sequence = row[0], row[seq_col_idx]
                if seq_id in ids_to_mutate:
                    for i in range(num_mutations_per_seq):
                        rate = np.random.normal(loc=0.10, scale=0.02)
                        mutated_seq = mutate_sequence(sequence, rate)
                        fasta_file.write(f">{seq_id}_{i}\n")
                        fasta_file.write(f"{mutated_seq}\n")
        print(f"Successfully created mutated FASTA file at: {output_fasta_path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error during mutated FASTA creation: {e}")


# --- 2. Core ESM Model and Embedding Logic ---

def load_esm_model(
    model_name_or_path: str, 
    base_model_name: str = None,
    is_mtl_model: bool = False,
    is_lora_model: bool = False, # <--- ADDED
    device: torch.device = None
) -> Tuple[torch.nn.Module, esm.Alphabet]:
    """
    Loads an ESM model. It can load a base model by name, a fully fine-tuned model,
    an MTL model, or a LoRA-tuned MTL model.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")

    # Case 1: Loading a fine-tuned model from a checkpoint file
    if os.path.isfile(model_name_or_path):
        if not base_model_name:
            raise ValueError("`base_model_name` must be provided when loading a fine-tuned checkpoint.")
        
        # --- NEW SECTION: Handles LoRA-tuned MTL Models ---
        if is_lora_model:
            print(f"Loading LoRA-tuned MTL model from checkpoint: {model_name_or_path}")
            
            # Step 1: Create base MTL model
            mtl_model = MTLModel(esm_name=base_model_name, head_hidden=512)
            
            # Step 2: Apply PEFT LoRA structure
            modules_to_save = ["ec_head", "kcat_head", "ph_head", "temp_head"]
            peft_config = LoraConfig(
                r=16, 
                lora_alpha=32, 
                target_modules=['q_proj', 'v_proj', 'k_proj'], 
                bias="none", 
                modules_to_save=modules_to_save
            )
            lora_model = get_peft_model(mtl_model, peft_config)
            print("Applied LoRA PEFT configuration.")

            # Step 3: Load weights with intelligent key fixing
            checkpoint = torch.load(model_name_or_path, map_location="cpu")
            state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
            
            fixed_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                if new_key.endswith('.lora_A.weight'):
                    new_key = new_key.replace('.lora_A.weight', '.lora_A.default.weight')
                elif new_key.endswith('.lora_B.weight'):
                    new_key = new_key.replace('.lora_B.weight', '.lora_B.default.weight')
                if not new_key.startswith('base_model.model.'):
                    new_key = 'base_model.model.' + new_key
                fixed_state_dict[new_key] = value

            lora_model.load_state_dict(fixed_state_dict, strict=False)
            print("Successfully loaded LoRA weights with key fixing.")
            
            # Step 4: Extract the underlying ESM encoder for embedding
            model = lora_model.base_model.model.encoder.esm
            alphabet = lora_model.base_model.model.alphabet
        
        # --- This part is your previous logic for non-LoRA models ---
        else:
            print(f"Loading standard fine-tuned model from checkpoint: {model_name_or_path}")
            print(f"Using base model architecture: {base_model_name}")
            
            # Load base architecture
            model, alphabet = esm.pretrained.load_model_and_alphabet(base_model_name)
            
            if not is_mtl_model:
                # For standard fine-tuned models
                checkpoint = torch.load(model_name_or_path, map_location=device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded standard fine-tuned weights.")
            else:
                # For MTL models (without LoRA)
                mtl_model = MTLModel(esm_name=base_model_name, head_hidden=512, dropout=0.1)
                checkpoint = torch.load(model_name_or_path, map_location=device)
                state_dict = checkpoint.get('model')
                mtl_model.load_state_dict(state_dict, strict=False)
                model = mtl_model.encoder.esm
                alphabet = mtl_model.encoder.alphabet
                print(f"Loaded MTL model's ESM encoder from {model_name_or_path}")

    # Case 2: Loading a pre-trained base model by its official name
    else:
        print(f"Loading pre-trained base model from ESM library: {model_name_or_path}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name_or_path)

    model = model.to(device)
    model.eval()
    return model, alphabet

def generate_embeddings(
    model: torch.nn.Module,
    alphabet: esm.Alphabet,
    fasta_path: str,
    output_dir: str,
    batch_size: int = 32,
    repr_layer: int = 33
):
    """
    Generates and saves embeddings for all sequences in a FASTA file.
    (This function is unchanged)
    """
    if not os.path.exists(fasta_path):
        print(f"FASTA file not found: {fasta_path}. Skipping.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    batch_converter = alphabet.get_batch_converter()

    try:
        data = list(esm.data.read_fasta(fasta_path))
    except Exception as e:
        print(f"Could not read FASTA file {fasta_path}: {e}")
        return

    print(f"Processing {len(data)} sequences from {fasta_path}...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc=f"Generating Embeddings for {os.path.basename(fasta_path)}"):
            batch_data = data[i : i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(next(model.parameters()).device)

            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            representations = results["representations"][repr_layer]

            for j, (label, seq_str) in enumerate(zip(batch_labels, batch_strs)):
                seq_id = label.split()[0]
                output_path = os.path.join(output_dir, f"{seq_id}.pt")
                
                if os.path.exists(output_path):
                    continue

                seq_len = len(seq_str)
                seq_repr = representations[j, 1 : seq_len + 1].mean(0)
                
                result = {"label": seq_id, "mean_representations": seq_repr.cpu()}
                torch.save(result, output_path)
                
    print(f"Embeddings for {os.path.basename(fasta_path)} saved in {output_dir}")


# --- 3. Main Execution Block ---

def main():
    config = {
        # --- CHOOSE YOUR MODEL ---
        # Path to a fine-tuned checkpoint (.pth file).
        "model_name_or_path": "/home/fangchh/workdir/enzyme/mtl/data/artifacts/best_model_epoch_8.pth",
        
        # Set to True if the checkpoint is from a LoRA training.
        "is_lora_model": True, # <--- CHANGED/ADDED
        
        # Set to True if it's a non-LoRA MTL model. (Set to False if is_lora_model is True).
        "is_mtl_model": False, # <--- CHANGED
        
        # Base model architecture. Required for any fine-tuned checkpoint.
        "base_model_name": "esm2_t33_650M_UR50D",

        # --- I/O AND DATA ---
        "fasta_files": [
            "./data/split100.fasta",
            "./data/price.fasta",
            "./data/new.fasta"
        ],
        "output_dir": "./data/esm2_lora_embeddings", # Renamed for clarity
        
        # --- DATA AUGMENTATION ---
        "run_mutation": True,
        "csv_file": "./data/split100.csv",
        
        # --- INFERENCE PARAMETERS ---
        "batch_size": 32,
        "repr_layer": 33
    }

    # --- Step 1: Optional Data Augmentation ---
    if config["run_mutation"]:
        if not config["csv_file"]:
            print("Error: 'csv_file' must be provided in config when 'run_mutation' is True.")
            return
        
        mutated_fasta_path = os.path.join(os.path.dirname(config["csv_file"]), "mutated_sequences.fasta")
        create_mutated_fasta(
            csv_path=config["csv_file"],
            output_fasta_path=mutated_fasta_path,
            embedding_dir=config["output_dir"]
        )
        if os.path.exists(mutated_fasta_path) and mutated_fasta_path not in config["fasta_files"]:
            config["fasta_files"].append(mutated_fasta_path)

    # --- Step 2: Load Model ---
    try:
        model, alphabet = load_esm_model(
            model_name_or_path=config["model_name_or_path"], 
            base_model_name=config["base_model_name"], 
            is_mtl_model=config["is_mtl_model"],
            is_lora_model=config["is_lora_model"] # <--- Pass the new flag
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading model: {e}")
        return

    # --- Step 3: Generate Embeddings ---
    for fasta_file in config["fasta_files"]:
        generate_embeddings(
            model=model,
            alphabet=alphabet,
            fasta_path=fasta_file,
            output_dir=config["output_dir"],
            batch_size=config["batch_size"],
            repr_layer=config["repr_layer"]
        )

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()