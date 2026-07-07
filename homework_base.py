from langchain_ollama import OllamaEmbeddings
import os, dotenv

dotenv.load_dotenv()
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def get_demensions():
    # 임베딩 객체
    embedding = OllamaEmbeddings(
        model='embeddinggemma:latest',
        base_url=os.getenv('OLLAMA_BASE_URL')
    )
    # Ollama에 임베딩 요청
    test_vector = embedding.embed_query("Hello")
    return len(test_vector)

