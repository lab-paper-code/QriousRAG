import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import faiss
import numpy as np
from datasets import load_dataset
from rouge import Rouge
from tqdm import tqdm
import json


# 1. Tokenizer 및 임베딩 모델 로드 (LLM2Vec)
model_embed = AutoModel.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
tokenizer_embed = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
# padding token이 없어서 eos_token을 padding token으로 설정
tokenizer_embed.pad_token = tokenizer_embed.eos_token
# Quantization 설정 (4-bit)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# Quantized 인코더 모델 로드
model_embed = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    quantization_config=quantization_config,
    device_map="auto"  # 자동으로 적절한 디바이스 할당
)

# 모델을 명시적으로 .to(device)로 옮길 필요 없음, 이미 올바른 디바이스로 할당됨
model_embed.eval()  # 평가 모드로 전환

# 폴더 경로 설정
folder_path = "./web_cleared_30"

# corpus 리스트 초기화
corpus = []

# 폴더 내에서 .txt 파일만 선택하여 내용을 corpus 리스트에 저장
for file_name in os.listdir(folder_path):
    # .txt 파일만 처리
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            # 파일 내용을 읽어서 corpus 리스트에 추가
            corpus.append(file.read())
            
            
# JSON 로드 함수
def load_triviaqa_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 압축 해제된 JSON 파일 경로 (실제 JSON 파일 이름에 맞게 수정 필요)
json_file_path = 'verified-web-dev.json'

# TriviaQA 데이터셋 로드
triviaqa_data = load_triviaqa_json(json_file_path)

# 데이터 전처리
# Queries & Correct answers
questions = [item["Question"] for item in triviaqa_data["Data"][:30]]
answers = [item["Answer"]["Aliases"] for item in triviaqa_data["Data"][:30]]

print("Questions loaded:", len(questions))
print("Answers loaded:", len(answers))

def embed(texts):
    # 텍스트를 토큰화하고 패딩 및 잘림 처리
    inputs = tokenizer_embed(texts, return_tensors="pt", padding=True, truncation=True)
    
    # 토큰화된 텍스트를 GPU로 보내기
    inputs = {key: value.to(model_embed.device) for key, value in inputs.items()}
    
    # 모델을 이용해 임베딩 계산
    with torch.no_grad():
        outputs = model_embed(**inputs)
        # 인코더의 마지막 히든 상태를 평균하여 임베딩 벡터 생성
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings

corpus_embeddings = embed(corpus)
# FAISS 인덱스 생성 및 문서 추가
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# 2. Tokenizer 및 텍스트 생성 모델 로드 (Meta-Llama-3.1)
tokenizer_gen = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model_gen = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# padding token이 없으면 eos_token을 padding token으로 설정
tokenizer_gen.pad_token = tokenizer_gen.eos_token

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_gen.to(device)
model_gen.eval()    # 생성 모델을 평가 모드로 전환


# 7. 평가 결과 저장용 변수 초기화
total_rouge_1 = 0
total_rouge_2 = 0
total_rouge_l = 0

num_samples = len(corpus)

def combine_query_and_docs(query, retrieved_docs):
    context_dict = {
        "Question": query,
        "Context": {
            "Retrieved_Articles": {f"Document_{i+1}": doc for i, doc in enumerate(retrieved_docs)}
        },
        "Instructions": {
            "Criteria": {
                "Faithfulness": "Is the generated answer factually accurate and aligned with the information from the retrieved documents?",
                "Relevance": "How relevant is the answer to both the query and the retrieved documents?",
                "Diversity": "Does the generated answer use a rich and varied vocabulary, or is it repetitive?",
                "Coherence": "Is the generated answer logically structured and easy to understand, maintaining a natural flow?"
            },
            "Task": "Create a single sentence that meets the four criteria above, and evaluate the following:",
            "Output_Format": {
                "Generated_Sentence": "A single sentence that meets the criteria.",
                "Scores": {
                    "Faithfulness": "Score from 0-10 based on how well the sentence meets the Faithfulness criterion.",
                    "Relevance": "Score from 0-10 based on how well the sentence meets the Relevance criterion.",
                    "Diversity": "Score from 0-10 based on how well the sentence meets the Diversity criterion.",
                    "Coherence": "Score from 0-10 based on how well the sentence meets the Coherence criterion."
                },
                "Loss": "Calculate each score's loss as (10 - score), and output the results in Python dictionary format.",
                "Average_Score": "Calculate the average score across all criteria."
            },
        }
    }
        # 딕셔너리에서 프롬프트로 변환
    prompt = (
        f"Question: {context_dict['Question']}\n"
        f"The following context is retrieved from relevant articles:\n"
        + "\n".join([f"{doc_id}: {doc}" for doc_id, doc in context_dict['Context']['Retrieved_Articles'].items()]) + "\n"
        "Based on the above context, please evaluate the generated sentence according to the following criteria:\n"
        + "\n".join([f"{criterion}: {description}" for criterion, description in context_dict['Instructions']['Criteria'].items()]) + "\n"
        f"{context_dict['Instructions']['Task']}\n"
        f"output format: {context_dict['Instructions']['Output_Format']}\n"
    )
    return prompt

def generate_answer(query, retrieved_docs, model_gen):
    # 프롬프트 불러오기
    input_text = combine_query_and_docs(query, retrieved_docs)

    # 입력 텍스트를 토크나이즈하고 모델로 생성 요청 (Meta-Llama-3.1 모델 사용)
    inputs = tokenizer_gen(input_text, return_tensors="pt", truncation=True).to(device)

    # 명시적으로 attention_mask를 추가
    inputs['attention_mask'] = (inputs['input_ids'] != tokenizer_gen.pad_token_id).long().to(device)

    # 답변 생성
    with torch.no_grad():  # 그래디언트 계산 비활성화
        generated = model_gen.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,  # attention_mask 추가
            pad_token_id=tokenizer_gen.pad_token_id,  # pad_token_id를 명시적으로 설정
            max_new_tokens=1024,
        )
    
    generated_text = tokenizer_embed.decode(generated[0], skip_special_tokens=True)
    # 생성된 텍스트 디코딩
    generated_text = tokenizer_embed.decode(generated[:, inputs.input_ids.shape[1]:][0], skip_special_tokens=True)
    
    # 텍스트에서 필요한 정보 추출
    generated_answer = {}
    try:
        generated_answer['Generated_Sentence'] = generated_text.split("'Generated_Sentence': '")[1].split("', 'Scores': ")[0]
        generated_answer['Scores'] = eval(generated_text.split("', 'Scores': ")[1].split(", 'Loss': ")[0])  # 딕셔너리 형태로 변환
        generated_answer['Loss'] = eval(generated_text.split(", 'Loss': ")[1].split(", 'Average_Score': ")[0])
        generated_answer['Average_Score'] = float(generated_text.split(", 'Average_Score': ")[1].split("}")[0])
    except (IndexError, ValueError):
        print("Error in parsing the generated text.")
        return None
    
    return generated_answer

# 정확도 계산을 위한 변수 초기화
correct_answers = 0

# 8. 성능 평가 루프 - 데이터셋 전체 순회
for i in tqdm(range(num_samples), desc="Evaluating"):
    query = corpus[i]
    reference_answer = answers[i] if len(answers[i]) > 0 else ""  # 첫 번째 정답 사용
    
    # 정답이 없으면 건너뜀
    if not reference_answer:
        continue
    
    # 9. 쿼리 임베딩 계산
    query_embedding = embed_batch([query], batch_size=2)

    # 10. FAISS에서 가장 가까운 문서 검색
    D, I = index.search(query_embedding, k=5)  # 가장 가까운 5개의 문서 검색

    # 11. 검색된 문서 출력
    retrieved_docs = [corpus[idx] for idx in I[0]]

    generated_text = generate_answer(query, retrieved_docs, model_gen)
    
    # 생성된 텍스트가 None이면 건너뜀
    if generated_text is None:
        print(f"Skipping sample {i} due to generation error.")
        continue
    
    print(f"Question : {query}")
    print(f"Generated Answer : {generated_text['Generated_Sentence']}")
    print(f"Original Answer : {reference_answer}")
    
    #  정답 포함 여부 확인
    if reference_answer.lower() in generated_text['Generated_Sentence'].lower():
        print(f"Sample {i}: Correct")
        correct_answers += 1
    else:
        print(f"Sample {i}: Incorrect")
    print("----------------------------------")
    # ROUGE 점수 계산
    rouge = Rouge()
    rouge_scores = rouge.get_scores(str(generated_text['Generated_Sentence']), reference_answer)
    total_rouge_1 += rouge_scores[0]['rouge-1']['f']
    total_rouge_2 += rouge_scores[0]['rouge-2']['f']
    total_rouge_l += rouge_scores[0]['rouge-l']['f']

# 평균 점수 계산 및 출력
avg_rouge_1 = total_rouge_1 / num_samples
avg_rouge_2 = total_rouge_2 / num_samples
avg_rouge_l = total_rouge_l / num_samples

# 정확도 계산
accuracy = correct_answers / num_samples

print(f"Average ROUGE-1: {avg_rouge_1:.4f}")
print(f"Average ROUGE-2: {avg_rouge_2:.4f}")
print(f"Average ROUGE-L: {avg_rouge_l:.4f}")

print(f"Accuracy: {accuracy:.4f}")
