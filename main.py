from sentence_transformers import SentenceTransformer
import numpy
import faiss

sentences = [
    "The weather is absolutely beautiful and sunny today.",
    "It is a gorgeous, bright, and cloudless day outside.",
    "I love eating pepperoni pizza with extra cheese for dinner.",
    "My favorite evening meal is a warm slice of cheese pizza.",
    "The young boy trained hard to win the local chess tournament.",
    "A kid practiced chess every day to secure victory in the competition.",
    "How much does it cost to fix a broken smartphone screen?",
    "I need to know the price for repairing a cracked phone display.",
    "The stock market experienced a sudden drop in early trading.",
    "Quantum computing relies on the principles of superposition."
]

## 모델 생성 (SentenceTransformer 모델 로드)
model = SentenceTransformer('bert-base-nli-mean-tokens')

# 문장들을 임베딩 (벡터로 변환)
sentence_embeddings = model.encode(sentences)

# 검색할 문장을 임베딩
query_text = "I just broken my iphone screen."
query_embeddings = model.encode([query_text])

# 유사도 계산
result = cosine_similarity(query_embeddings, sentence_embeddings)
scores = result[0]

# 점수 출력
for i, s in enumerate(scores):
    print(f"{i}. {sentences[i]}: {s}")