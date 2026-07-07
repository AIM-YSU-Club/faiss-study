from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss, os, dotenv

from homework_base import sentences, get_demensions

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
query_text = "내 스마트폰 화면이 부서졌습니다." # 검색할 문장
results = faiss_vdb.similarity_search_with_score(query_text, 3)
# 검색 결과를 출력
for i, r in enumerate(results):
    document = r[0]
    score = r[1]
    print(f"검색 결과 {i + 1}: {document.page_content}")
    print(f"유사도 점수 (코사인): {score}")