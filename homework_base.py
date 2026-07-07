from langchain_ollama import OllamaEmbeddings
import os, dotenv

dotenv.load_dotenv()
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sentences = [
    # wjs 벡터DB 문장
    "제주도 여행 중 방문한 그 카페는 통유리창 너머로 바다가 한눈에 보여서 인상적이었다.",
    "바다가 정면으로 내려다보이는 통창 덕분에 그 제주 카페는 특히 기억에 남았다.",
    "새로 산 노트북은 배터리가 하루 종일 가고 발열도 거의 없어서 만족스럽다.",
    "이번에 구입한 노트북은 발열이 적고 배터리 지속 시간도 길어서 마음에 든다.",
    "회사 프로젝트 마감이 다가오면서 팀원들 모두 야근을 반복하고 있다.",
    "마감 기한이 임박하자 팀 전체가 매일 늦게까지 남아서 일하고 있다.",
    "아침마다 조깅을 하고 나면 하루 종일 컨디션이 훨씬 좋아지는 걸 느낀다.",
    "매일 아침 달리기를 마치고 나면 몸 상태가 확실히 좋아지는 게 느껴진다.",
    "그 다큐멘터리는 심해 생물들의 놀라운 생존 전략을 다루고 있었다.",
    "중앙은행이 기준금리를 동결하기로 결정하면서 시장이 안도하는 분위기다."
    # 여기서부터 원하는 문장들을 추가하세요.
]

# 벡터 차원 수 구하는 함수
# FAISS 벡터DB 객체 만들 때 필요합니다.
def get_demensions():
    # 임베딩 객체
    embedding = OllamaEmbeddings(
        model='embeddinggemma:latest',
        base_url=os.getenv('OLLAMA_BASE_URL')
    )
    # Ollama에 임베딩 요청
    test_vector = embedding.embed_query("Hello")
    return len(test_vector)

