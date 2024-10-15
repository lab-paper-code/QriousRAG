import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the pre-trained SentenceTransformer model
# GPU가 가능하면 자동으로 GPU로 모델을 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


dataset=pd.read_csv('/raid/deallab/SF_RAG_Data/ASQA/embedding_train.csv')
# Convert the dataset to the SentenceTransformer input format
def prepare_data(data, text_column1, text_column2):
    """
    CSV 파일에서 두 개의 텍스트 열을 `SentenceTransformer`의 InputExample 형식으로 변환
    :param data: 데이터프레임
    :param text_column1: 첫 번째 텍스트 열 (예: 질문)
    :param text_column2: 두 번째 텍스트 열 (예: 문서/컨텍스트)
    :return: InputExample 리스트
    """
    # 데이터에서 NaN 값을 빈 문자열로 대체하고, 비문자열을 제거
    data[text_column1] = data[text_column1].fillna('').astype(str)
    data[text_column2] = data[text_column2].fillna('').astype(str)

    input_examples = []
    for _, row in data.iterrows():
        text1 = row[text_column1]
        text2 = row[text_column2]
        input_examples.append(InputExample(texts=[text1, text2]))

    return input_examples

train_data = prepare_data(dataset, 'question', 'text')

data=dataset[['question','text']]
question=data['question'].tolist()
context=data['text'].tolist()

train_batch_size = 32
# 데이터 전처리
train_data = prepare_data(dataset, 'question', 'text')

# Train/Val split
train_df, val_df = train_test_split(dataset[['question', 'text']], test_size=0.3, random_state=42)

# 학습용 데이터셋 준비
train_examples = prepare_data(train_df, 'question', 'text')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)

# 검증용 데이터셋 준비
val_examples = prepare_data(val_df, 'question', 'text')

# Loss 설정 (MultipleNegativesRankingLoss)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Early Stopping을 위한 설정
patience = 3
best_val_loss = float('inf')
patience_counter = 0

# 검증을 위한 evaluator 설정 (EmbeddingSimilarityEvaluator 사용)
val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='validation')

# Train the model with GPU support
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,  # 조정 가능
    evaluator=val_evaluator,
    show_progress_bar=True,
    use_amp=True  # 혼합 정밀도(FP16) 사용, 필요에 따라 생략 가능
)
# 학습된 모델 저장
model.save("fine_tuned_model")

def search_documents(query, documents, model, device="cuda"):
    # 질문 임베딩 생성 (GPU로 이동)
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    # 문서 임베딩 생성 (GPU로 이동)
    document_embeddings = model.encode(documents, convert_to_tensor=True).to(device)
    
    # 질문과 문서 간 코사인 유사도 계산 (PyTorch 기반)
    query_embedding = query_embedding.unsqueeze(0)  # 배치 차원 추가
    similarities = torch.nn.functional.cosine_similarity(query_embedding, document_embeddings)

    # 유사도에 따라 문서 정렬
    top_results = similarities.argsort(descending=True)[:20]
    print(top_results)
    res={}
    for idx in top_results:
        tmp=documents[idx]
        res[tmp]=1
        
    return list(res.keys())

# 예시 사용법
for i in range(10):
    print(f"Query {i+1} : {question[i]}")
    print("-"*100)
    results = search_documents(question[i], context[i], model)
    print("Retrieved doc :")
    for j in range(len(results)):
        print(f"\tRank {j} : {results[j]}")
    print("Original doc :", context[i], "\n")