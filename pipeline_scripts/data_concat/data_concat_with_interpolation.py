#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shared_utils.status_manager as status_manager


# In[ ]:


warnings.filterwarnings("ignore")


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess headset CSV files into fixed-size Numpy arrays.")

    defaults = {
        "input_dir": "data/zengo_recording",
        "output_dir": "data/zengo_preprocessed",
        "target_length": 2000,
        "target_filename": "headset.csv",
        "overwrite": False
    }

    # Required Arguments
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=defaults['input_dir'], 
        help=f"Path to the root directory containing raw CSV files (default: { defaults['input_dir'] })"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=defaults['output_dir'], 
        help=f"Path to the root directory where .npy files will be saved (default: { defaults['output_dir'] })"
    )

    # Optional Arguments (Configurable defaults)
    parser.add_argument(
        "--target_length", 
        type=int, 
        default=defaults['target_length'], 
        help=f"The fixed length to resize the time series to (default: { defaults['target_length'] })"
    )

    parser.add_argument(
        "--target_filename", 
        type=str, 
        default=defaults['target_filename'], 
        help=f"The specific CSV filename to search for (default: { defaults['target_filename'] })"
    )

    parser.add_argument(
        "--overwrite", 
        action="store_true",
        default=defaults['overwrite'], 
        help=f"If set, overwrites existing .npy files (default: { defaults['overwrite'] })"
    )

    return parser.parse_args()


# In[3]:


def resize_data_to_numpy(df, target_length):
    numeric_df = df.select_dtypes(include=[np.number])

    # NOTE: Ha van olyan oszlop amire nincs szükség
    # (pl.: időbélyeg) azt még itt kell eldobni

    data = numeric_df.values # (n_sor, n_oszlop)
    original_length = data.shape[0]

    if original_length < 2:
        return None

    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)

    f = interp1d(x_old, data, axis=0, kind="linear")
    data_resized = f(x_new)
    return data_resized.astype(np.float32)


# In[4]:


def main():
    args = parse_args()

    # --- SETUP ---
    if not os.path.exists(args.input_dir):
        print(f"Error: The input directroy '{args.input_dir}' does not exsist.")
        sys.exit(1)

    print(f"Starting preprocessing...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target: {args.target_filename} -> Resized to {args.target_length} rows (with interpolation)")

    # --- COLLECT FILES ---
    files_to_process = []
    for root, dirs, files in os.walk(args.input_dir):
        if args.target_filename in files:
                files_to_process.append(os.path.join(root, args.target_filename))

    print(f"Found {len(files_to_process)} files to process.")
    print("-" * 50)

    # --- PREPROCESSING ---
    total_files = len(files_to_process)
    print(f"Starting preprocessing on {total_files} files...")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, file_path in enumerate(tqdm(files_to_process, desc="Preprocessing")):
        if total_files > 0:
            percent_done = int((i / total_files) * 100)
            status_manager.update_progress("Concatenation", percent_done)

        relative_path = os.path.relpath(file_path, args.input_dir)

        relative_save_path = os.path.splitext(relative_path)[0] + ".npy"

        save_path = os.path.join(args.output_dir, relative_save_path)

        # Skip check
        if os.path.exists(save_path) and not args.overwrite:
            skipped_count += 1
            continue

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            df = pd.read_csv(file_path)

            df = df.interpolate(method="linear").bfill().ffill()

            processed_data = resize_data_to_numpy(df, args.target_length)

            if processed_data is None:
                print(f"Warning: Data empty or too short in {relative_path}!")
                error_count += 1
                continue

            np.save(save_path, processed_data)
            processed_count += 1

        except Exception as e:
            print(f"Error in file: {file_path} -> {e}")

    status_manager.update_progress("Concatenation", 100)

    print("-" * 50)
    print(f"DONE!")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




