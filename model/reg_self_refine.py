import os
import importlib
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import torch
from datasets import Dataset
import pandas as pd
from collections import defaultdict
import re
import numpy as np
import time
from evaluation import evaluate
import prompts
importlib.reload(prompts)

'''
nohup python3 rag_self_refine.py > out.txt
'''

device1 = 'cuda:0'
device2 = 'cuda:1'

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

data=qa_df[['question','long_answers']]
questions=data['question']

references = [row.to_dict() for i, row in qa_df.iterrows() if i < len(questions)]

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

def evaluate_docs(query, docs):
    print(f"Query : {query}")
    print("-"*100)
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
        
        filter=(generated_text.split('\n')[0])
        print(filter)
        if '#relevant' in filter:
            outs.append((generated_text.split('\n')[1]).strip())
    
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
        {"role":"user", 'content':{input}},
    ]
    inputs = tokenizer_gen.apply_chat_template(messages, return_tensors="pt", truncation=True).to(device1)
    
    attention_mask = (inputs != tokenizer_gen.pad_token_id).long().to(device1)
    
    outputs = model_gen.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens=256)
    generated_text = tokenizer_gen.decode(outputs[0]).split('<|end_header_id|>')[-1].replace('<|eot_id|>', '').strip('\n')
    
    return generated_text

def preprocessing(new_questions):
    return list(((new_questions.split("['")[1]).split("']")[0]).split("',\n    '"))

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
        {"role":"user", 'content':{input}},
    ]
    inputs = tokenizer_gen.apply_chat_template(messages, return_tensors="pt", truncation=True).to(device1)
    
    attention_mask = (inputs != tokenizer_gen.pad_token_id).long().to(device1)
    
    outputs = model_gen.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens=256)
    generated_text = tokenizer_gen.decode(outputs[0]).split('<|end_header_id|>')[-1].replace('<|eot_id|>', '').strip('\n')
    
    return generated_text

def final_ans(query,answers):
    prompt = f"""
    Context information is below.
    ---------------------
    {answers}
    ---------------------
    Given the context information and not prior knowledge, 
    Answer the question that have multiple correct answers based on multiple interpretations, including multiple answers.
    Query: {query}
    Answer:
    """
    
    input_ids = tokenizer_gen.apply_chat_template([{"role":'user', "content":prompt}], return_tensors='pt').to(device1)

    attention_mask = (input_ids != tokenizer_gen.pad_token_id).long().to(device1)

    out = model_gen.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer_gen.pad_token_id, max_new_tokens = 512)
    res = tokenizer_gen.decode(out[0]).split('<|end_header_id|>')[-1] 
    candidate = [re.sub('\n|<\|eot_id\|>', '', res)]
    return candidate

sf_rag=dict()
perplexity_df=pd.DataFrame()
scores_list=[]
new_answers_dic=dict()

for i in range(300):
    print(f"Query {i+1} : {questions[i]}")
    print("-"*100)
    query = questions[i]
    new_answers_dic[query]=list()
    rel_docs = retrieve_documents(query)
    if rel_docs:
        new_questions=preprocessing(make_new_query(query, rel_docs))
    else:
        print("Pass to the next query")
        continue
    print(new_questions)
    for new_question in new_questions:
        new_docs=retrieve_documents(new_question)
        new_rel_docs=evaluate_docs(new_question, new_docs)
        if new_rel_docs:
            new_answers=make_new_answer(new_question, new_rel_docs)
            print(new_answers)
            new_answers_dic[query].append(new_answers)
        else:
            print('All irrelevant docs')
    candidate=final_ans(query,new_answers_dic[query])
    print(candidate)
    print(references[i])
    scores=evaluate(candidate,[references[i]])
    print(scores)
    scores_list.append(scores)
    scores_df=pd.DataFrame(scores_list)
    scores_df.mean()
    scores_df.to_csv(f'./results/QRiousRAG_results.csv', index=False)