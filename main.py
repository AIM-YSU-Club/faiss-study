from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document

import faiss
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# 검색할 문장
query_text = "내 스마트폰 화면이 부서졌습니다."

# 임베딩 객체
embedding = OllamaEmbeddings(
    model='embeddinggemma:latest',
    base_url='203.230.208.82:8000'
)
# Ollama에 임베딩 요청
test_vector = embedding.embed_query("Hello")
d = len(test_vector)

# LangChain 인터페이스로 FAISS 벡터DB 생성
faiss_vdb = FAISS(
    embedding_function=embedding,
    index=faiss.IndexFlatIP(d),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
# 벡터 DB에 문장 추가 -> (임베딩 자동 수행)
faiss_vdb.add_texts(sentences)

faiss_vdb.save_local('./faissdata/', 'vdb')

# 벡터 DB에서 유사도 검색
results = faiss_vdb.similarity_search_with_score(query_text, 3)
# 검색 결과를 출력
for r in results:
    document = r[0]
    score = r[1]
    print(f"검색 결과 1: {document.page_content}")
    print(f"유사도: {score}")