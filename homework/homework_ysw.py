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
query_text = "나는 자기 전에 휴대폰을 보는 습관이 있다." # 검색할 문장
results = faiss_vdb.similarity_search_with_score(query_text, 3)
# 검색 결과를 출력
for i, r in enumerate(results):
    document = r[0]
    score = r[1]
    print(f"검색 결과 {i + 1}: {document.page_content}")
    print(f"유사도 점수 (코사인): {score}")

## 검색 결과 ##
"""
검색 결과 1: 나는 자기 전에 휴대폰을 보는 습관이 있다. 망했다.
유사도 점수 (코사인): 0.8261533975601196
검색 결과 2: 나는 밤에 코딩하는 것이 더 집중이 잘된다.
유사도 점수 (코사인): 0.4953186511993408
검색 결과 3: 나는 백수다.
유사도 점수 (코사인): 0.49418017268180847

"""