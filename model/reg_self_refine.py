import os
import importlib
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
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
from tqdm import tqdm

data_dir = '/raid/deallab/SF_RAG_Data/ASQA'
data_dir = '../data'

device1 = 'cuda:0'
device2 = 'cuda:1'

from evaluation import evaluate
import prompts
importlib.reload(prompts)

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
        
    return top_results, res

def evaluate_docs(query, docs):
    # print(f"Query : {query}")
    # print("-"*100)
    outs = []
    for idx, doc in enumerate(docs):
        #print(f"Rank {idx} : {doc}")
        
        input= f'''
        Query: {query}
        Doc:  {doc}
        '''

        messages = [
            {"role":"user", 'content':prompts.PROMPT['eval_doc_instr']},
            {"role":"assistant", 'content':prompts.PROMPT['eval_doc_answ1']},
            {"role":"user", 'content':prompts.PROMPT['eval_doc_ex2']},
            {"role":"assistant", 'content':prompts.PROMPT['eval_doc_answ2']},
            {"role":"user", 'content':prompts.PROMPT['eval_doc_ex3']},
            {"role":"assistant", 'content':prompts.PROMPT['eval_doc_answ3']},
            {"role":"user", 'content':input}, 
        ]
        #apply tokenizter + generate eval
        inputs = tokenizer_gen.apply_chat_template(messages, return_tensors="pt", truncation=True).to(device1)
        
        attention_mask = (inputs != tokenizer_gen.pad_token_id).long().to(device1)
        
        outputs = model_gen.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens=128)
        generated_text = tokenizer_gen.decode(outputs[0]).split('<|end_header_id|>')[-1].replace('<|eot_id|>', '').strip('\n')
        
        if '#relevant' in generated_text:
            outs.append(doc)
    
    return outs

def make_new_query(query,context):
    
    input= f'''
    Original Query: {query}
    Context information: {context}
    '''
    
    messages = [
        {"role":"user", 'content':prompts.PROMPT['refine_query_instr']},
        {"role":"assistant", 'content':prompts.PROMPT['refine_query_answ1']},
        {"role":"user", 'content':prompts.PROMPT['refine_query_ex2']},
        {"role":"assistant", 'content':prompts.PROMPT['refine_query_answ2']},
        {"role":"user", 'content':input},
    ]
    inputs = tokenizer_gen.apply_chat_template(messages, return_tensors="pt", truncation=True).to(device1)
    
    attention_mask = (inputs != tokenizer_gen.pad_token_id).long().to(device1)
    
    outputs = model_gen.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens=256)
    generated_text = tokenizer_gen.decode(outputs[0]).split('<|end_header_id|>')[-1].replace('<|eot_id|>', '').strip('\n')
    generated_text  = generated_text.strip('[]').split(',\n')
    
    print(generated_text)
    return generated_text

def make_new_answer(query,context):
    
    input= f'''
    Original Query: {query}
    Context information: {context}
    '''
    
    messages = [
        {"role":"user", 'content':prompts.PROMPT['new_answer_instr']},
        {"role":"assistant", 'content':prompts.PROMPT['new_answer_answ1']},
        {"role":"user", 'content':prompts.PROMPT['new_answer_ex2']},
        {"role":"assistant", 'content':prompts.PROMPT['new_answer_answ2']},
        {"role":"user", 'content':input},
    ]
    inputs = tokenizer_gen.apply_chat_template(messages, return_tensors="pt", truncation=True).to(device1)
    
    attention_mask = (inputs != tokenizer_gen.pad_token_id).long().to(device1)
    
    outputs = model_gen.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens=256)
    generated_text = tokenizer_gen.decode(outputs[0]).split('<|end_header_id|>')[-1].replace('<|eot_id|>', '').strip('\n')
    
    # print(f'New Answer: {generated_text}')
    return generated_text

def final_ans(query,answer, qa_pairs):
    # prompt = f"""
    # Context information is below.
    # ---------------------
    # {answers}
    # ---------------------
    # Given the context information and not prior knowledge, 
    # Answer questions that have multiple correct answers based on multiple interpretations, including multiple answers.
    # Query: {query}
    # Answer:
    # """
    
    # input_ids = tokenizer_gen.apply_chat_template([{"role":'user', "content":prompt}], return_tensors='pt').to(device1)
    qa_sample = '''Follow-up Query{i}: {q}
    Context: {context}
    '''
    qa_string = '\n'.join([qa_sample.format(i=i, q=q, context=a) for i, (q, a) in enumerate(qa_pairs)])
    
    input= f'''
    Initial Query: {query}
    Context: {answer}
    {qa_string}
    '''
    print(f'Final Answ Input:{input}')
    messages = [
        {"role":"user", 'content':prompts.PROMPT['final_answer_instr']},
        {"role":"assistant", 'content':prompts.PROMPT['final_answer_answ1']},
        {"role":"user", 'content':input},
    ]

    #tokenizer prompt
    input_ids = tokenizer_gen.apply_chat_template(messages, return_tensors="pt", truncation=True).to(device2)

    attention_mask = (input_ids != tokenizer_gen.pad_token_id).long().to(device1)

    out = model_gen.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens = 512)
    res = tokenizer_gen.decode(out[0]).split('<|end_header_id|>')[-1] 
    candidate = [re.sub('\n|<\|eot_id\|>', '', res)]
    #print(candidate)
    return candidate

from evaluation import evaluate
from collections import defaultdict

# sf_rag=dict()
# perplexity_df=pd.DataFrame()
scores_list=[]
stop_iteration = 5
new_answers_dic=defaultdict(list)

for idx, row in tqdm(qa_df.iterrows(), total=min(len(qa_df), stop_iteration)):
    if idx == stop_iteration: break
    query = row['question']
    
    #retrieve relevant docs
    ids, docs = retrieve_documents(query)
    rel_docs = evaluate_docs(query, docs)
    answer = make_new_answer(query, rel_docs)
    print(f'Initial Answer: {answer}')
    
    # generate new queries
    new_queries=make_new_query(query, rel_docs)
    
    #iterate over new docs
    qa_pairs = []
    for i, new_query in tqdm(enumerate(new_queries)):
        if i == 5: break #brak after x follow-up question 
        
        # retrieve relevant docs
        ids, new_docs=retrieve_documents(new_query)
        new_rel_docs=evaluate_docs(new_query, new_docs)
        new_answer = make_new_answer(new_query, new_rel_docs)
        qa_pairs.append((new_query, new_answer))

    # generate final answer
    candidate=final_ans(query, answer, qa_pairs)
    print(f'candidate: {candidate}')
    # print(references[i])
    scores=evaluate(candidate,[row.to_dict()])
    print(scores)
    scores_list.append(scores)
    
scores_df=pd.DataFrame(scores_list)
print(scores_df.mean())
scores_df.to_csv('./results/self-refine_results.csv', index=False)