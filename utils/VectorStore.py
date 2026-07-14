from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss, os, dotenv

# .env파일 읽어오기
dotenv.load_dotenv()

class VectorStore:
    def __init__(self):
      
      self.embedding_model = OllamaEmbeddings(
      model='embeddinggemma:latest',
      base_url=os.getenv('OLLAMA_BASE_URL')
      )
      
      self.faiss_vdb = FAISS(
      embedding_function=self.embedding_model, # 임베딩 객체
      index=faiss.IndexFlatIP(self.get_d()),     # 차원 수
      docstore=InMemoryDocstore(),               
      index_to_docstore_id={}
      )
      
    # 벡터 차원 수 구하기
    def get_d(self):
      # Ollama에 임베딩 요청
      test_vector = self.embedding_model.embed_query("Hello")
      return len(test_vector)
    
    # 벡터DB 생성 및 저장
    def create(self, sentences: list[str]):
      self.faiss_vdb.add_texts(sentences)
      self.faiss_vdb.save_local(os.getenv('FAISS_DATA_PATH'), 'faiss_study')
    
    def search(self, query: str):
      results = self.faiss_vdb.similarity_search_with_score(query, 3)
      result_logs = []
      for i, r in enumerate(results):
          document = r[0]
          score = r[1]
          result_logs.append(f"검색 결과 {i + 1}: {document.page_content}")
          result_logs.append(f"유사도 점수 (코사인): {score}")
          
      return "\n".join(result_logs)
        