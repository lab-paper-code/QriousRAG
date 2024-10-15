import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import faiss
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json

# Tokenizer 및 Meta-Llama-3.1-8B-Instruct 인코더 모델 로드
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

# JSON 로드 함수
def load_triviaqa_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 압축 해제된 JSON 파일 경로 (실제 JSON 파일 이름에 맞게 수정 필요)
json_file_path = 'verified-web-dev.json'

# TriviaQA 데이터셋 로드
triviaqa_data = load_triviaqa_json(json_file_path)

filenames = []
questions=[]
answers=[]
for qa in triviaqa_data["Data"]:
    for file in qa["SearchResults"]:
        questions.append(qa["Question"])
        answers.append(qa["Answer"]["Aliases"])
        filename = file["Filename"]
        if filename not in filenames:
            filenames.append(filename)

print("Questions loaded:", len(questions))
print("Answers loaded:", len(answers))
print("File length:", len(filenames))

# 폴더 경로 설정
folder_path = "./triviaqa_corpus"

# corpus 리스트 초기화
corpus = []

# 폴더 내에서 .txt 파일만 선택하여 내용을 corpus 리스트에 저장
for file_name in os.listdir(folder_path):
    # .txt 파일만 처리
    print(file_name)
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            # 파일 내용을 읽어서 corpus 리스트에 추가
            corpus.append(file.read())
            

def embed_batch(texts, batch_size):
    all_embeddings = []
    
    # 데이터를 배치 단위로 나누어 처리
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 텍스트를 토큰화하고 패딩 및 잘림 처리
        inputs = tokenizer_embed(batch_texts, return_tensors="pt", padding=True, truncation=True)
        
        # 토큰화된 텍스트를 GPU로 보내기
        inputs = {key: value.to(model_embed.device) for key, value in inputs.items()}
        
        # 모델을 이용해 임베딩 계산
        with torch.no_grad():
            outputs = model_embed(**inputs)
            # 인코더의 마지막 히든 상태를 평균하여 임베딩 벡터 생성
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # 각 배치의 임베딩을 리스트에 저장
        all_embeddings.append(embeddings)
    
    # 모든 배치의 임베딩을 하나의 numpy array로 병합
    return np.vstack(all_embeddings)

# 배치 처리로 임베딩 계산
corpus_embeddings = embed_batch(corpus, batch_size=4)

# 저장된 corpus_embeddings 불러오기
loaded_embeddings = np.load("corpus_embeddings_50.npy")
print("Embeddings loaded successfully")

# FAISS 인덱스 생성 및 문서 추가
corpus_embeddings=loaded_embeddings
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# 2. Tokenizer 및 텍스트 생성 모델 로드 (Meta-Llama-3.1)
tokenizer_gen = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model_gen = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"  # 자동으로 적절한 디바이스 할당
)

# padding token이 없으면 eos_token을 padding token으로 설정
tokenizer_gen.pad_token = tokenizer_gen.eos_token

model_gen.eval()    # 생성 모델을 평가 모드로 전환

def embed_query(query):
    # 쿼리 임베딩 계산
    inputs = tokenizer_embed([query], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(model_embed.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model_embed(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    return query_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine_query_and_docs(query, retrieved_docs, previous_answer=None):
    # Context 딕셔너리 생성
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

    # 만약 previous_answer가 있으면, 평가된 결과를 포함하여 보완 요청
    if previous_answer:
        context_dict["Instructions"]["Previous_Answer"] = {
            "Generated_Sentence": previous_answer["Generated_Sentence"],
            "Scores": previous_answer["Scores"],
            "Loss": previous_answer["Loss"],
            "Feedback": f"Your previous answer has the following evaluation: Faithfulness: {previous_answer['Scores']['Faithfulness']}, "
                        f"Relevance: {previous_answer['Scores']['Relevance']}, Diversity: {previous_answer['Scores']['Diversity']}, "
                        f"Coherence: {previous_answer['Scores']['Coherence']}. "
                        f"Please improve the generated answer by addressing the areas with the highest loss values. The goal is to reduce these losses."
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

    if previous_answer:
        # 이전 답변과 피드백을 포함
        prompt += (
            f"Previous Generated Sentence: {previous_answer['Generated_Sentence']}\n"
            f"Scores: {previous_answer['Scores']}\n"
            f"Loss: {previous_answer['Loss']}\n"
            f"Please improve the generated sentence by addressing the feedback provided based on the Loss values.\n"
        )

    return prompt

def generate_answer(query, retrieved_docs, model_gen, previous_answer):
    # 프롬프트 불러오기
    input_text = combine_query_and_docs(query, retrieved_docs, previous_answer)

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

def evaluate_answer(answer):
    """LLM이 생성한 답변에 대해 평가 지표에 따른 점수를 반환."""
    sentence=answer['Generated_Sentence']
    scores=answer['Scores']
    loss = answer['Loss']
    average_score = sum(scores.values()) / len(scores)

    return sentence, scores, loss, average_score

def optimize_answer(query, retrieved_docs, model_gen, max_iterations=10, loss_threshold=1, patience=3):
    """Loss 값을 줄여가며 최적의 답변을 생성"""
    best_loss = float('inf')
    best_sentence = None
    best_scores = None
    best_average_score = 0
    no_improvement_counter = 0  # 개선되지 않은 반복 수를 셀 카운터
    iteration = 0
    previous_answer = None  # 이전에 생성된 답변을 저장할 변수
    
    # 최적의 답변을 찾기 위해 Loss가 가장 작은 답변을 반복하여 찾기
    while iteration < max_iterations:
        
        # 답변 생성
        answer = generate_answer(query, retrieved_docs, model_gen, previous_answer)
        
        if answer is None:
            print("Error generating answer. Generate answer again.")
            continue
        
        print(f"Iteration {iteration + 1}:")
        
        # 답변 평가
        sentence, scores, loss, average_score = evaluate_answer(answer)
        total_loss = sum(loss.values())
        
        print(f"Generated sentence: {sentence}")
        print(f"Scores: {scores}")
        print(f"Loss: {loss}")
        print(f"Total Loss: {total_loss}")
        print(f"Average Score: {average_score}")
        print(">>")

        # Loss가 더 나아졌으면 최적의 답변으로 설정
        if total_loss < best_loss:
            best_sentence = sentence
            best_scores = scores
            best_loss = total_loss
            best_average_score = average_score
            no_improvement_counter = 0  # 개선이 이루어졌으므로 카운터 초기화
        else:
            no_improvement_counter += 1  # 개선되지 않았으면 카운터 증가

        # Loss가 기준 이하로 작으면 최적의 답변으로 간주하고 반복 종료
        if total_loss <= loss_threshold:
            print("Loss가 충분히 줄어들었습니다. 최적의 답변을 찾았습니다.")
            break

        # patience만큼 개선이 없으면 반복 종료
        if no_improvement_counter >= patience:
            print(f"Loss 개선이 {patience}회 동안 이루어지지 않았습니다. 반복을 종료합니다.")
            break
        
        # 이전 답변을 저장하여 다음 반복에서 사용
        previous_answer = answer
        iteration += 1

    print(f"Best Answer: {best_sentence}")
    print(f"Best Scores: {best_scores}")
    print(f"Best Loss: {best_loss}")
    print(f"Best Average Score: {best_average_score:.2f}")
    
    return best_sentence  # 최적의 답변을 반환

# 정확도 계산을 위한 변수 초기화
correct_answers=0
generated_cnt=0

num_samples = len(questions)

# 8. 성능 평가 루프 - 데이터셋 전체 순회
for i in tqdm(range(num_samples), desc="Evaluating"):
    query = questions[i]
    reference_answer = answers[i] if len(answers[i]) > 0 else ""  # 첫 번째 정답 사용
    
    # 정답이 없으면 건너뜀
    if not reference_answer:
        continue
    print(f"Question : {query}")
    print(f"Original Answer : {reference_answer}")
    # 9. 쿼리 임베딩 계산
    query_embedding = embed_query(query)

    # 10. FAISS에서 가장 가까운 문서 검색
    D, I = index.search(query_embedding, k=2)  # 오히려 k를 줄일수록 성능 개선 -> 아마도 많은 후보군이 존재할 경

    # 11. 검색된 문서 출력
    retrieved_docs = [corpus[idx] for idx in I[0]]
    print(f"Retrieved_docs : {retrieved_docs}")
    
    best_answer = optimize_answer(query, retrieved_docs, model_gen, max_iterations=10, loss_threshold=1, patience=3)
    # generated_text = generate_answer(query, retrieved_docs, model_gen)
    
    # 생성된 텍스트가 None이면 건너뜀
    if best_answer is None:
        print(f"Skipping sample {i} due to generation error.")
        continue
    else:
        generated_cnt+=1
    
    print(f"Generated Answer : {best_answer}")
    #  정답 포함 여부 확인
    c=0
    for answer in reference_answer:
        if answer.lower() in best_answer.lower():
            print(f"Sample {i}: Correct")
            correct_answers += 1
            c=1
            break
    if c==0:
        print(f"Sample {i}: Incorrect")
    print("----------------------------------")

# 정확도 계산
accuracy = correct_answers / generated_cnt
print("generated_cnt",generated_cnt)
print(f"Accuracy: {accuracy:.4f}")