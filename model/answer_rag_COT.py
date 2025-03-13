import random, os
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import pandas as pd
import re

device1 = 'cuda:0'
data_dir = '/raid/deallab/SF_RAG_Data/ASQA'
# data_dir = '../data'

def set_seed(seed_value):
    # Set seed for reproducibility.
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic=True    
    torch.backends.cudnn.benchmark=True
    torch.cuda.manual_seed_all(seed_value)
    
#load embeddings
embedd_test_path = f'{data_dir}/test/embedd_test.npy'
evidence_embeddings = np.load(embedd_test_path)
print(evidence_embeddings.shape)
evidence_embeddings = torch.from_numpy(evidence_embeddings).to(device1)

#load embeddings (evidence_eval)
embedd_test_path = f'{data_dir}/test/embedd_test_eval2.npy'
evidence_embeddings_eval = np.load(embedd_test_path)
print(evidence_embeddings_eval.shape)
evidence_embeddings_eval = torch.from_numpy(evidence_embeddings_eval).to(device1)

#load embeddings (evidence_title)
embedd_test_path = f'{data_dir}/test/embedd_test_eval_title.npy'
evidence_embeddings_title = np.load(embedd_test_path)
print(evidence_embeddings_title.shape)
evidence_embeddings_title = torch.from_numpy(evidence_embeddings_title).to(device1)

#load evidence
evidence_test_path = f'{data_dir}/test/evidence_test2.csv'
evidence_df = pd.read_csv(evidence_test_path)
print(len(evidence_df))

#load evidence
evidence_test_path = f'{data_dir}/test/evidence_test_eval.csv'
evidence_eval_df = pd.read_csv(evidence_test_path)
print(len(evidence_eval_df))

#load title evidence
evidence_test_path = f'{data_dir}/test/evidence_test_eval_title.csv'
evidence_eval_title_df = pd.read_csv(evidence_test_path)
print(len(evidence_eval_title_df))

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
# cache_dir= '/raid/deallab/.cache')
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
    device_map= 'auto',
    # cache_dir= '/raid/deallab/.cache'
)
model_gen.eval()

evidence_eval_path = f'{data_dir}/test/evidence_test_eval.csv'
eval_df = pd.read_csv(evidence_eval_path)
eval_df.head(20)

def CoT(query, context):
    prompt = """
    Context information is below.
    ——————————
    {0}
    ——————————
    In context, there may be content A, B, C, etc. for an ambiguous question. 
    Let's create a long answer that includes various related content such as A, B, C, etc.
    Query: {1}
    Answer: Let's think about it step by step.
    """.format('\n'.join(context), query)
    input_ids = tokenizer_gen.apply_chat_template([{"role":'user', "content":prompt}], return_tensors='pt').to(device1)

    attention_mask = (input_ids != tokenizer_gen.pad_token_id).long().to(device1)

    out = model_gen.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens = 512)
    res = tokenizer_gen.decode(out[0]).split('<|end_header_id|>')[-1] 
    return re.sub('\n|<\|eot_id\|>', '', res)

def answer(query, context):
    prompt = """
    Context information is below.
    ---------------------
    {0}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    
    Query: {1}
    Answer: Let's think step by step.
    """.format('\n'.join(context), query)
    input_ids = tokenizer_gen.apply_chat_template([{"role":'user', "content":prompt}], return_tensors='pt').to(device1)

    attention_mask = (input_ids != tokenizer_gen.pad_token_id).long().to(device1)

    out = model_gen.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens = 512)
    res = tokenizer_gen.decode(out[0]).split('<|end_header_id|>')[-1] 
    return re.sub('\n|<\|eot_id\|>', '', res)

# retrive docs from the document embeddings
# retrive docs from the document embeddings
def retrieve_documents(query,num=10):
    max_length = 1024
    
    #query prefix
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    
    query_embedding = model.encode([query],instruction=query_prefix, max_length=max_length).to(device1)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, evidence_embeddings)

    top_results = similarities.argsort(descending=True)[:num].cpu().detach().numpy()
    res=[evidence_df.loc[idx, 'text'] for idx in top_results if idx < len(evidence_df)]
    idx=[idx for idx in top_results if idx < len(evidence_df)]
    return res, idx


from tqdm import tqdm
from evaluation import evaluate

set_seed(24)

stop_iteration = len(qa_df)
retrival_list=[]
scores_list=[]
for idx, row in tqdm(qa_df.iterrows(), total=min([stop_iteration, len(qa_df)])):
    if idx == stop_iteration: break
    query = row['question']
    retrieved_docs, doc_ids = retrieve_documents(query,10)
    tmp=0

    ans=CoT(query,retrieved_docs)
    print('Final ans:', ans)
    scores=evaluate([ans], [row.to_dict()])
    scores_list.append(scores)
    scores_df=pd.DataFrame(scores_list)
    scores_df.to_csv('./results/COT2_1000_results.csv', index=False)
    print(scores)

        
scores_df=pd.DataFrame(scores_list)
print(scores_df.mean())

scores_df.to_csv('./results/COT2_1000_results.csv', index=False)
import pandas as pd
import math
sf = pd.read_csv('results/COT2_1000_results.csv')
sf=sf[sf['length']<1000][:]
print(len(sf))
print(sf.mean())
r=sf.mean()['rougeLsum']
d=sf.mean()['Disambig-F1']
dr=math.sqrt(r*d)
print(dr)
