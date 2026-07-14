from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss, os, dotenv

from homework_base import sentences, get_demensions

dotenv.load_dotenv()

# 1. 임베딩 모델 객체 생성

embedding = OllamaEmbeddings(
    model = 'embeddinggemma:latest',
    base_url = os.getenv('OLLAMA_BASE_URL')
)

# 2. faiss 벡터 DB 생성

faiss_vdb = FAISS(
    embedding_function = embedding,              
    index = faiss.IndexFlatIP(get_demensions()),
    docstore = InMemoryDocstore(),
    index_to_docstore_id = {}

)

# 3. 벡터 DB에 문장들 추가

faiss_vdb.add_texts(sentences)

# 4. 유사도 검색

query_text = "오늘은 날씨가 좋다."
result = faiss_vdb.similarity_search_with_score(query_text, 3)

# 5. 검색 결과 출력

for i, r in enumerate(result):
    document = r[0]
    score = r[1]
    print(f"{1 + i}번째로 유사한 문장 : {document.page_content}")
    print(f"유사도 점수 (코사인) : {score}")

# 6. 유사도 검색 결과

# 1번째로 유사한 문장 : 하늘에 구름 한 점 없이 맑고 따뜻한 날씨입니다.
# 유사도 점수 (코사인) : 0.6797984838485718
# 2번째로 유사한 문장 : 기온이 올라가면서 완연한 봄 날씨가 느껴집니다.
# 유사도 점수 (코사인) : 0.5625845789909363
# 3번째로 유사한 문장 : 이번 여름 휴가는 시원한 바다로 떠나고 싶네요.
# 유사도 점수 (코사인) : 0.4526553153991699