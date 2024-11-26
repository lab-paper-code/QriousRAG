import os
import importlib
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import torch
#from sentence_transformers import SentenceTransformer, InputExample, losses
#from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
import numpy as np
import time

'''
nohup python3 rag_baseline.py > out.txt
'''

device1 = 'cuda:0'
device2 = 'cuda:1'

data_dir = '/raid/deallab/SF_RAG_Data/ASQA'
data_dir = '../data'

#load embeddings
embedd_test_path = f'{data_dir}/test/embedd_test.npy'
evidence_embeddings = np.load(embedd_test_path)
print(evidence_embeddings.shape)
evidence_embeddings = torch.from_numpy(evidence_embeddings).to(device1)

#load evidence
evidence_test_path = f'{data_dir}/test/evidence_test.csv'
evidence_df = pd.read_csv(evidence_test_path)

#load qa data
qa_df=pd.read_csv(f'{data_dir}/test/qa_test.csv') #data=df[['question','long_answers']] # questions=data['question'] #references = [row.to_dict() for i, row in df.iterrows() if i < len(questions)]
qa_df.head()

#load quantized model
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
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage =True,
)
model.eval()

#load tokenizer
tokenizer_gen = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer_gen.pad_token = tokenizer_gen.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_storage=torch.bfloat16,
)

model_gen = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map= 'auto'
)
model_gen.eval()

# retrive docs from the document embeddings
def retrieve_documents(query):
    max_length = 1024
    
    #query prefix
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    
    query_embedding = model.encode([query],instruction=query_prefix, max_length=max_length).to(device1)

    # query_embedding = query_embedding.unsqueeze(0)
    #print(query_embedding)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, evidence_embeddings)

    top_results = similarities.argsort(descending=True)[:10].cpu().detach().numpy()
    #print(top_results)
    res=[evidence_df.loc[idx, 'text'] for idx in top_results if idx < len(evidence_df)]
        
    return res

from evaluation import evaluate
from tqdm import tqdm
import nltk

stop_iteration = 2000

scores_list=[]
for idx, row in tqdm(qa_df.iterrows(), total=min([stop_iteration, len(qa_df)])):
    if idx == stop_iteration: break
    query = row['question']
    retrieved_docs = retrieve_documents(query)
    
    # print("Retrieved doc :")
    # for j in range(len(results)):
    #     print(f"\tRank {j} : {results[j]}")
        
    prompt = """
    Context information is below.
    ---------------------
    {0}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {1}
    Answer:
    """.format('\n'.join(retrieved_docs), query)
    input_ids = tokenizer_gen.apply_chat_template([{"role":'user', "content":prompt}], return_tensors='pt').to(device2)

    attention_mask = (input_ids != tokenizer_gen.pad_token_id).long().to(device2)

    out = model_gen.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens = 512)
    res = tokenizer_gen.decode(out[0]).split('<|end_header_id|>')[-1] 
    candidate = [re.sub('\n|<\|eot_id\|>', '', res)]

    scores=evaluate(candidate, [row.to_dict()])
    print(scores)
    scores_list.append(scores)
    scores_df=pd.DataFrame(scores_list)
    scores_df.to_csv(f'./results/baseline-11-25_results.csv', index=False)
    
scores_df=pd.DataFrame(scores_list)
print(scores_df.mean())
scores_df.to_csv(f'./results/baseline-11-25_results.csv', index=False)