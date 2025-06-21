import os
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_PATH = os.path.join("vectorstore", "db_faiss")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVER_SEARCH_K = 2

FLASK_API_URL = os.getenv("FLASK_API_URL")
