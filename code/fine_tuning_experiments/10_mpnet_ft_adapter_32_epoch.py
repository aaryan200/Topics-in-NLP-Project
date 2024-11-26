"""
Python script to rigorously finetune the TwoLayerNN adapter on top of the mpnet model.
Earlier, it was finetuned for 8 epochs. Now, we will finetune it for 32 epochs.
"""


import torch
from typing import Any, List, Optional, Tuple#, Union
from llama_index.core import SimpleDirectoryReader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
from llama_index.embeddings.huggingface.pooling import Pooling
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.finetuning.embeddings.adapter_utils import BaseAdapter
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
torch.manual_seed(0)


# ## Create dataset

# In[2]:


df_kb = pd.read_pickle('../../data/kb_chunks_emb.pkl')
print(df_kb.shape)
df_kb.head(3)


# In[3]:


df_ques_url_train = pd.read_pickle('../../data/questions_relevant_urls_chunks_train.pkl')
print(df_ques_url_train.shape)
df_ques_url_train.head(3)


# In[4]:


corpus = dict()
for _, row in df_kb.iterrows():
    if row['doc_url'] not in corpus:
        corpus[row['doc_url']] = row['chunk_content']


# In[5]:


train_queries = dict()
train_relevant_docs = dict()

for i, row in df_ques_url_train.iterrows():
    ques_id = str(i)
    rel_docs = row['relevant_docs_urls']
    train_queries[ques_id] = row['question']
    train_relevant_docs[ques_id] = rel_docs


# In[6]:


train_dataset = EmbeddingQAFinetuneDataset(
    queries = train_queries, corpus = corpus, relevant_docs = train_relevant_docs
)


# In[7]:


train_dataset.save_json('data/train_dataset.json')


# In[7]:


# requires torch dependency
from llama_index.embeddings.adapter.utils import TwoLayerNN
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.adapter import AdapterEmbeddingModel


# In[8]:


model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
base_embed_model = resolve_embed_model(f"local:{model_name}")


# In[9]:


adapter_model = TwoLayerNN(
    in_features=768,
    hidden_features=1024,
    out_features=768,
    bias=True,
    add_residual=True
)


# In[ ]:


finetune_engine = EmbeddingAdapterFinetuneEngine(
    train_dataset,
    base_embed_model,
    model_output_path="mpnet_finetuned_ep32",
    # model_checkpoint_path="model5_ck",
    adapter_model=adapter_model,
    epochs=32,
    verbose=False,
    device="cuda",
    batch_size = 32
)


# In[ ]:


import time

start = time.time()

finetune_engine.finetune()

end = time.time()

print(f"Time taken: {end-start}s")
print(f"Time taken per epoch: {(end-start)/32}s")

# In[ ]:




