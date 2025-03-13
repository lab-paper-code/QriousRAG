import random, os
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="3"

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
evidence_test_path = f'{data_dir}/test/evidence_test.csv'
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

def match_sample_id(query_id, doc_id):
    query_sample_id=qa_df.loc[query_id,'sample_id']
    doc_sample_id=eval_df.loc[doc_id,'sample_id']
    if query_sample_id==doc_sample_id:
        return 1
    else:
        return 0

def match_sample_id2(query_id, title):
    query_sample_id=qa_df.loc[query_id,'sample_id']
    doc_sample_id=evidence_eval_title_df.loc[evidence_eval_title_df['title']==title,'sample_id'].item()
    if query_sample_id==doc_sample_id:
        return 1
    else:
        return 0
    
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

# retrive docs from the document embeddings
def retrieve_documents_eval(query,num=10):
    max_length = 1024
    
    #query prefix
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    
    query_embedding = model.encode([query],instruction=query_prefix, max_length=max_length).to(device1)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, evidence_embeddings_eval)

    top_results = similarities.argsort(descending=True)[:num].cpu().detach().numpy()
    res=[evidence_eval_df.loc[idx, 'text'] for idx in top_results if idx < len(evidence_eval_df)]
    idx=[idx for idx in top_results if idx < len(evidence_eval_df)]
    return res, idx

# retrive docs from the document embeddings
def retrieve_documents2(query, embed, evidence, num=10):
    max_length = 1024
    
    #query prefix
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    
    query_embedding = model.encode([query],instruction=query_prefix, max_length=max_length).to(device1)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, embed)

    top_results = similarities.argsort(descending=True)[:num].cpu().detach().numpy()
    res=[evidence.loc[idx, 'text'] for idx in top_results if idx < len(evidence)]
    titles=[evidence.loc[idx, 'title'] for idx in top_results if idx < len(evidence)]
    # idx=[idx for idx in top_results if idx < len(evidence)]
    return res, titles

# retrive docs from the document embeddings
def retrieve_documents3(query, embed, evidence, num=10):
    max_length = 1024
    
    #query prefix
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    
    query_embedding = model.encode([query],instruction=query_prefix, max_length=max_length).to(device1)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, embed)

    top_results = similarities.argsort(descending=True)[:num].cpu().detach().numpy()
    res=[evidence.loc[idx, 'text'] for idx in top_results if idx < len(evidence)]
    # titles=[evidence.loc[idx, 'title'] for idx in top_results if idx < len(evidence)]
    idx=[idx for idx in top_results if idx < len(evidence)]
    return res, idx

# retrive docs from the document embeddings
def retrieve_titles(query,num=10):
    max_length = 1024
    
    #query prefix
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    
    query_embedding = model.encode([query],instruction=query_prefix, max_length=max_length).to(device1)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, evidence_embeddings_title)

    top_results = similarities.argsort(descending=True)[:num].cpu().detach().numpy()
    res=[evidence_eval_title_df.loc[idx, 'title'] for idx in top_results if idx < len(evidence_eval_title_df)]
    idx=[idx for idx in top_results if idx < len(evidence_eval_title_df)]
    return res, idx

def find_titles(query):
    titles,title_ids=retrieve_titles(query, 10)
    return titles

def find_docs(titles):
    docs=[]
    for title in titles:
        # print(title)
        # print(evidence_eval_df.loc[evidence_eval_df['title']==title,'text'].to_list())
        docs.extend(evidence_eval_df.loc[evidence_eval_df['title']==title,'text'])
        # print()
    return docs

def retrieve_inside(query, titles, num=10):
    docs=[]
    text_docs=pd.DataFrame(columns=['text','title'])
    final_docs=[]
    for title in titles:
        text_docs=pd.concat([text_docs, evidence_eval_df.loc[evidence_eval_df['title']==title,['text','title']]],ignore_index=True)
        docs.extend(evidence_embeddings_eval[evidence_eval_df['title']==title])
        # print(title_retrieved_docs)
        # print()
    docs_tensor = torch.stack(docs)
    title_retrieved_docs,t=retrieve_documents2(query, docs_tensor, text_docs, num)
    final_docs.extend(title_retrieved_docs)
    return final_docs,t

def answer_2020(query, title, context):
    prompt = """
    Context information is below.
    ---------------------
    {0}
    ---------------------
    Given the context information and not prior knowledge, answer this ambiuous question.
    1. You should include in answer the content according to the various interpretations of the question
    2. When using time-related information to answer the question, only use information before February 1, 2020.
    3. If you can’t answer the question, say "Not relevant" only.
    4. Title means topic of context, so it should be considered when making answer.
    5. Answers to questions should be no longer than 200 words.
    Do not comment your answer and strictly follow this instructions.
    Query: {1}
    Title: {2}
    Answer:
    """.format('\n'.join(context), query, title)
    input_ids = tokenizer_gen.apply_chat_template([{"role":'user', "content":prompt}], return_tensors='pt').to(device1)

    attention_mask = (input_ids != tokenizer_gen.pad_token_id).long().to(device1)

    out = model_gen.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens = 512)
    res = tokenizer_gen.decode(out[0]).split('<|end_header_id|>')[-1] 
    return re.sub('\n|<\|eot_id\|>', '', res)

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

stop_iteration = 20
retrival_list=[]
scores_list=[]
for idx, row in tqdm(qa_df.iterrows(), total=min([stop_iteration, len(qa_df)])):
    # if idx == stop_iteration: break
    query = row['question']   
    titles,title_ids=retrieve_titles(query, 10)
    # title_docs=find_docs(titles)
    docs=[]
    first_ans=[]
    for title in titles:
        # print(evidence_eval_df.loc[evidence_eval_df['title']==title,'text'].to_list())
        text_docs=evidence_eval_df.loc[evidence_eval_df['title']==title,'text'].to_list()
        docs_tensor=evidence_embeddings_eval[evidence_eval_df['title']==title]
        title_retrieved_docs,_=retrieve_documents3(query, docs_tensor, pd.DataFrame(text_docs, columns=['text']), 10)
        ans=answer_2020(query, title, title_retrieved_docs)
        if 'There is no relevant information' not in ans and len(ans) < 1000:
            print(ans)
            first_ans.append(ans)
        
    # final_docs, doc_titles=retrieve_inside(query, titles, 10)
    # print(final_docs, doc_titles)
    # # retrieved_docs, doc_ids = retrieve_documents2(query)
    # tmp=0
    # for doc_title in doc_titles:
    #     tmp+=match_sample_id2(idx, doc_title)
    # res=tmp/len(final_docs)
    # print("Retrival Match Rate:", res)
    # dic=dict()
    # dic['first_retrival']=res
    ans=total_answer(query,first_ans)
    # final_retrieved_documents,_ = retrieve_documents(ans)
    
    # final_retrieved_documents[5:10] = final_retrieved_documents[5:10][::-1]
    # ans=final_total_answer(query,final_retrieved_documents)
    print('Final ans:', ans)
    scores=evaluate([ans], [row.to_dict()])
    scores_list.append(scores)
    scores_df=pd.DataFrame(scores_list)
    print(scores)
    # retrival_list.append(dic)
    # retrival_df=pd.DataFrame(retrival_list)
    # print(dic)
        
    scores_df=pd.DataFrame(scores_list)
    scores_df.mean()
    scores_df.to_csv('./results/titleRAG200_02-19_results.csv', index=False)

# retrival_df=pd.DataFrame(retrival_list)
# retrival_df.mean()

scores_df.to_csv('./results/titleRAG200_02-19_results.csv', index=False)
import pandas as pd
import math
sf = pd.read_csv('results/titleRAG200_02-19_results.csv')
sf=sf[sf['length']<1000][:]
print(len(sf))
print(sf.mean())
r=sf.mean()['rougeLsum']
d=sf.mean()['Disambig-F1']
dr=math.sqrt(r*d)
print(dr)