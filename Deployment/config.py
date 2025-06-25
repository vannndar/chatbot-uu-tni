# File: config.py

import os
from dotenv import load_dotenv

# --- Konstanta yang tidak berubah ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(CURRENT_DIR, "vectorstore", "db_faiss")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVER_SEARCH_K = 2

# --- Fungsi untuk memuat konfigurasi API secara dinamis ---
def load_and_get_api_models():
    """
    Memuat ulang file .env dan membangun dictionary model API.
    Fungsi ini memastikan konfigurasi selalu terbaru pada setiap run.
    """
    # Menggunakan override=True untuk memaksa pembaruan variabel dari .env
    # jika sudah ada di lingkungan.
    load_dotenv(override=True)
    
    api_models = {}
    api_prefix = "API_URL_"

    for key, value in os.environ.items():
        if key.startswith(api_prefix):
            # Mengubah 'API_URL_MODEL_SAYA' menjadi 'Model Saya'
            model_name = key[len(api_prefix):].replace('_', ' ').title()
            if value: # Hanya tambahkan jika URL-nya tidak kosong
                api_models[model_name] = value
                
    return api_models