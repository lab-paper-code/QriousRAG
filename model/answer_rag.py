import random, os
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import pandas as pd
import re

device1 = 'cuda:0'
device2 = 'cuda:1'
data_dir = '/raid/deallab/SF_RAG_Data/ASQA'

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

def total_answer(query, docs):
    prompt = """
    Context information is below.
    ---------------------
    {0}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    1. The query is an ambiguous question.
    2. Therefore, you must include the contents according to the various interpretations of the query in one answer by utilizing the given context.
    3. Each content according to the various interpretations of the query must be explained in one or two sentences.
    4. The total answer must be 5 sentences or less.
    Do not comment your answer and strictly follow this instructions.
    Query: {1}
    Answer:
    """.format('\n'.join(docs), query)
    input_ids = tokenizer_gen.apply_chat_template([{"role":'user', "content":prompt}], return_tensors='pt').to(device1)

    attention_mask = (input_ids != tokenizer_gen.pad_token_id).long().to(device1)

    out = model_gen.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens = 512)
    res = tokenizer_gen.decode(out[0]).split('<|end_header_id|>')[-1] 
    return re.sub('\n|<\|eot_id\|>', '', res)

from tqdm import tqdm
from evaluation import evaluate

set_seed(24)

stop_iteration = 200

scores_list=[]
retrival_list=[]
for idx, row in tqdm(qa_df.iterrows(), total=min([stop_iteration, len(qa_df)])):
    if idx == stop_iteration: break
    query = row['question']
    
    retrieved_docs, doc_ids = retrieve_documents(query)
    # tmp1=0
    # for doc_id in doc_ids:    
    #     tmp1+=match_sample_id(idx, doc_id)
    # first_retrival=tmp1/len(retrieved_docs)
    # print("First Retrival Match Rate:", first_retrival)
    
    # dic=dict()
    # dic['first_retrival']=first_retrival
    
    first_ans=total_answer(query,retrieved_docs)
    print("First ans: ", first_ans)

    # virtual=[first_ans]
    # for _ in range(5):
    #     res=virtual_answer(query,first_ans)
    #     print(res)
    #     virtual.append(res)
    # virtual='. '.join(virtual)
    ans_docs, doc_ids=retrieve_documents(first_ans)
    final_ans=total_answer(query,ans_docs)
    
    scores=evaluate([final_ans], [row.to_dict()])
    scores_list.append(scores)
    scores_df=pd.DataFrame(scores_list)
    print(scores)
    scores_df.to_csv('./results/answer_rag_0219_seed24_results.csv', index=False)
    
    
    # retrival_list.append(dic)
    # retrival_df=pd.DataFrame(retrival_list)
    # print(dic)
        
scores_df=pd.DataFrame(scores_list)
scores_df.mean()
scores_df.to_csv('./results/answer_rag_0219_seed24_results.csv', index=False)


# retrival_df=pd.DataFrame(retrival_list)
# retrival_df.mean()