import json
import os
import random

SEED = 42
random.seed(42)

# load data
data = []
with open('/raid/deallab/SF_RAG_Data/N_Q/nq-train.jsonl', 'r') as f:
    for i, line in enumerate(list(f)):
        try:
            data.append(json.loads(line))
        except:
            print(f'Could not load line {i}')
            continue
    

len(data)

# sample from whole data
sample = random.sample(data, 10000)

# create evidence files
files = {}
for s in sample:
    id = s['example_id']
    text = s['document_text'].split()
    chunks = [(chunk['start_token'],chunk['end_token']) for chunk in s['long_answer_candidates'] if chunk['top_level']==True]
    for start, end in chunks:
        print(start, end)
        files[f'{id}_{start}_{end}'] = ' '.join(text[start, end])