#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


# In[2]:


# Config
RAW_DIR = "../zengo_recording"
PROCESSED_DIR = "../zengo_preprocessed"
TARGET_LENGTH = 2000 # Ez változhatna
TARGET_FILENAME = "headset.csv"


# In[3]:


def resize_data_to_numpy(df, target_length):
    numeric_df = df.select_dtypes(include=[np.number])

    # itt időbélyeg oszlopot eldobni

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


def preprocess_database():
    files_to_process = []
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file == TARGET_FILENAME:
                files_to_process.append(os.path.join(root, file))

    print(f"Feldolgozás indítása: {len(files_to_process)}db fájl...")

    for file_path in tqdm(files_to_process):
        try:
            df = pd.read_csv(file_path)

            df = df.interpolate(method="linear").bfill().ffill()

            processed_data = resize_data_to_numpy(df, TARGET_LENGTH)

            if processed_data is None:
                print("A feldolgozott df üres!")
                continue

            relative_path = os.path.relpath(file_path, RAW_DIR)
            relative_path = os.path.splitext(relative_path)[0] + ".npy"

            save_path = os.path.join(PROCESSED_DIR, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            np.save(save_path, processed_data)

        except Exception as e:
            print(f"HIBA ennél a fájlnál: {file_path} -> {e}")


# In[5]:


preprocess_database()


# In[ ]:




