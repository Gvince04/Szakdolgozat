#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
from torch.utils.data import Dataset
from momentfm import MOMENTPipeline
from transformers import Trainer, TrainingArguments

import gc
from tqdm import trange
import time
import os


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name":"embedding"}
)
model.init()
model.to("cuda").float()


# In[11]:


root_dir = "numpy_data"
embedding_result = []


for file in os.listdir(root_dir):
    if file.endswith(".npy"):
        data = np.load(f"{root_dir}/{file}")
        print(data.shape)
    else:
        continue

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    data = data.T.unsqueeze(0)

    chunk_size = 2048
    all_embeddings = []

    for start in trange(0, data.shape[2], chunk_size):
        end = min(start + chunk_size, data.shape[2])
        chunk = data[:, :, start:end].to("cuda")
        chunk_mask = torch.ones(
            chunk.shape[0],
            chunk.shape[2],
            dtype=bool, 
            device="cuda"
        )

        with torch.no_grad():
            out = model(x_enc=chunk, input_mask=chunk_mask)
            embeddings = out.embeddings.cpu()
            all_embeddings.append(embeddings)

        del chunk, chunk_mask, out, embeddings
        torch.cuda.empty_cache()
        gc.collect()

    final_embeddings = torch.cat(all_embeddings, dim=0)

    embedding = final_embeddings.mean(dim=0)
    embedding_result.append(embedding)


# In[ ]:


print(embedding_result)


# In[3]:


scaler = StandardScaler()
data = scaler.fit_transform(data)
print(data.shape)
data[0]


# In[4]:


model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name":"embedding"}
)
model.init()
model.to("cuda").float()


# In[13]:


if isinstance(data, np.ndarray):
    data = torch.tensor(data, dtype=torch.float32)

data.shape


# In[ ]:


input_mask = ~torch.isnan(data)

print(input_mask.shape)
input_mask[0, :]


# In[ ]:


data = data.T.unsqueeze(0) # since for MOMENT the data has to be [batch, features, sequence]


# In[14]:


# for i in trange(10, desc="Processing"):
#     time.sleep(0.5)


# In[16]:


chunk_size = 2048
all_embeddings = []

for start in trange(0, data.shape[2], chunk_size):
    end = min(start + chunk_size, data.shape[2])
    chunk = data[:, :, start:end].to("cuda")
    chunk_mask = torch.ones(
        chunk.shape[0],
        chunk.shape[2],
        dtype=bool, 
        device="cuda"
    )

    with torch.no_grad():
        out = model(x_enc=chunk, input_mask=chunk_mask)
        embeddings = out.embeddings.cpu()
        all_embeddings.append(embeddings)

    del chunk, chunk_mask, out, embeddings
    torch.cuda.empty_cache()
    gc.collect()

final_embeddings = torch.cat(all_embeddings, dim=0)


# In[17]:


print(final_embeddings.shape)


# In[18]:


embedding = final_embeddings.mean(dim=0)


# In[19]:


embedding[:15]


# In[ ]:




