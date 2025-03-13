import pandas as pd
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig,  BitsAndBytesConfig
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

'''run embedding

nohup python3 embedd_test_evidence.py > out.txt

'''

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# load model with tokenizer
model = AutoModel.from_pretrained(
    'nvidia/NV-Embed-v2', 
    trust_remote_code=True,
    quantization_config = bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
model.eval()

batchsize = 8
max_len = 1024

data_folder = '/raid/deallab/SF_RAG_Data/ASQA'
data_folder = '../data'

# read evidence data
evidence_test_path = f'{data_folder}/test/evidence_test.csv'
evidence_df = pd.read_csv(evidence_test_path)

#dump directory
embedd_test_path = f'{data_folder}/test/embedd_test.npy'
evidence_embeddings = []

#prepare data
evidence_ds = Dataset.from_pandas(evidence_df)
evidence_dl = DataLoader(evidence_ds, batch_size=batchsize, shuffle=False)
for evidence in tqdm(evidence_dl):
    evidence_embeddings.append(model.encode(evidence['text'], max_length = max_len).cpu().detach().numpy())
    
# turn to np arr
evidence_embeddings = np.concatenate(evidence_embeddings, axis=0)
print(evidence_embeddings.shape)
np.save(embedd_test_path, evidence_embeddings)