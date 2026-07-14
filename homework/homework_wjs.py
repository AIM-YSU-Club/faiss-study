from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss, os, dotenv

from homework.homework_base import sentences, get_demensions

dotenv.load_dotenv()

# 임베딩 객체
embedding = OllamaEmbeddings(
    model='embeddinggemma:latest',
    base_url=os.getenv('OLLAMA_BASE_URL')
)

# LangChain 인터페이스로 FAISS 벡터DB 생성
faiss_vdb = FAISS(
    embedding_function=embedding,              # 임베딩 객체
    index=faiss.IndexFlatIP(get_demensions()),   # 차원 수
    docstore=InMemoryDocstore(),               
    index_to_docstore_id={}
)

# 벡터 DB에 문장 추가 -> (임베딩 자동 수행)
faiss_vdb.add_texts(sentences)
# 유사도 검색 수행
query_text = "아침마다 조깅을 하는 습관은 좋은 습관이다." # 검색할 문장
results = faiss_vdb.similarity_search_with_score(query_text, 3)
# 검색 결과를 출력
for i, r in enumerate(results):
    document = r[0]
    score = r[1]
    print(f"검색 결과 {i + 1}: {document.page_content}")
    print(f"유사도 점수 (코사인): {score}")

## 검색 결과 ##
"""
검색 결과 1: 아침마다 조깅을 하고 나면 하루 종일 컨디션이 훨씬 좋아지는 걸 느낀다.
유사도 점수 (코사인): 0.7456008791923523
검색 결과 2: 매일 아침 달리기를 마치고 나면 몸 상태가 확실히 좋아지는 게 느껴진다.
유사도 점수 (코사인): 0.5916985869407654
검색 결과 3: 새로 산 노트북은 배터리가 하루 종일 가고 발열도 거의 없어서 만족스럽다.
유사도 점수 (코사인): 0.28454092144966125
"""