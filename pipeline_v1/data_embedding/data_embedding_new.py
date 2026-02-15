#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import sys
import gc
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from momentfm import MOMENTPipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MOMENT embeddings for headset.npy files.")

    # Required Arguments
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="../zengo_preprocessed", 
        help="Path to the root directory containing preprocessed .npy files (default: '../zengo_preprocessed')"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../zengo_embedded", 
        help="Path to the root directory where embeddings will be saved (default: ../zengo_embedded)"
    )

    # Optional Arguments (with defaults)
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="AutonLab/MOMENT-1-large", 
        help="HuggingFace model ID (default: AutonLab/MOMENT-1-large)"
    )

    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=2048, 
        help="Context window size for MOMENT (default: 2048)"
    )

    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="If set, overwrites existing embedding files."
    )

    return parser.parse_args()


# In[ ]:


def main():
    args = parse_args()

    # --- SETUP & CHECKS ---
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting embedding generation on {device.upper()}...")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # --- LOAD MODEL ---
    print(f"Model:  {args.model_name}")
    try:
        model = MOMENTPipeline.from_pretrained(
            args.model_name,
            model_kwargs={"task_name": "embedding"}
        )
        model.init()
        model.to(device).float() # To cuda if available, else cpu
        model.eval()

    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # --- PROCESSING LOOP ---
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Collect all files first to show a progress bar (tqdm)
    files_to_process = []
    for root, dirs, files in os.walk(args.input_dir):
        if "headset.npy" in files:
            files_to_process.append(os.path.join(root, "headset.npy"))

    print(f"Found {len(files_to_process)} files to process.")
    print("-" * 50)

    # tqdm progress bar
    for input_path in tqdm(files_to_process, desc="Embedding"):

        # Calculate paths
        relative_path = os.path.relpath(os.path.dirname(input_path), args.input_dir)

        # Create output folder
        output_folder = os.path.join(args.output_dir, relative_path)
        output_path = os.path.join(output_folder, "headset_embedded.npy")

        # Skip if exists and not overwriting
        if os.path.exists(output_path) and not args.overwrite:
            skipped_count += 1
            continue

        try:
            os.makedirs(output_folder, exist_ok=True)

            # --- LOAD DATA ---
            raw_data = np.load(input_path)

            # --- PREPROCESS ---
            # 1. Handle NaNs
            if np.isnan(raw_data).any():
                raw_data = np.nan_to_num(raw_data, nan=0.0)

            # 2. Standard Scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(raw_data)

            # 3. Reshape: (Time, Channels) -> (Batch, Channels, Time)
            # Transpose + Unsqueeze: (2000, 126) -> (1, 126, 2000)
            data_tensor = torch.tensor(scaled_data, dtype=torch.float32).T.unsqueeze(0)

            # --- INFERENCE ---
            data_tensor = data_tensor.to(device)

            # Create Mask
            input_mask = torch.ones(
                data_tensor.shape[0], 
                data_tensor.shape[2], 
                dtype=torch.long, 
                device=device
            )

            with torch.no_grad():
                output = model(x_enc=data_tensor, input_mask=input_mask)
                embeddings = output.embeddings.cpu()

            # --- SAVE OUTPUT ---
            # Average over channels -> (1024,)
            final_embedding = embeddings.mean(dim=1).squeeze(0).numpy()
            np.save(output_path, final_embedding)

            processed_count += 1

            # --- CLEANUP ---
            # Delete unusied variables
            del data_tensor, input_mask, output, embeddings, raw_data, scaled_data

            # torch.cuda.empty_cache() # This can be deleted since the input data is always a shape of (2000, 126)

            # Forces garbage collector to collect
            gc.collect()

        except Exception as e:
            tqdm.write(f"Error processing {relative_path}: {e}")
            error_count += 1

    print("-" * 50)
    print(f"   DONE!")
    print(f"   Processed: {processed_count}")
    print(f"   Skipped:   {skipped_count}")
    print(f"   Errors:    {error_count}")

if __name__ == "__main__":
    main()

