# File: setup_vectorstore.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- PERUBAHAN DI SINI ---
from langchain_huggingface import HuggingFaceEmbeddings
# --- ----------------- ---
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader

# Muat environment variable dari file .env
load_dotenv()

# Path ke data dan vector store
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Fungsi untuk membuat vector store
def create_vector_store():
    # 1. Load Dokumen
    print("Memuat dokumen...")
    # Anda menyebutkan file ini, jadi saya gunakan ini sebagai contoh
    # Load PDF documents
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    documents = []
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
        pdf_docs = loader.load()
        documents.extend(pdf_docs)
        print(f"PDF dokumen {pdf_file} berhasil dimuat: {len(pdf_docs)} halaman.")
    
    # Load JSON documents
    
    json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
    
    for json_file in json_files:
        try:
            # Using JSONLoader with jq-style content extraction
            loader = JSONLoader(
                file_path=os.path.join(DATA_PATH, json_file),
                jq_schema='.', # Extract all content (adjust as needed)
                text_content=False
            )
            json_docs = loader.load()
            documents.extend(json_docs)
            print(f"JSON dokumen {json_file} berhasil dimuat: {len(json_docs)} item.")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Total dokumen berhasil dimuat: {len(documents)}")

    # 2. Split Teks menjadi Chunks
    print("Membagi teks menjadi chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Jumlah chunks: {len(docs)}")

    # 3. Buat Embeddings (MENGGUNAKAN MODEL GRATIS DARI HUGGING FACE)
    print("Membuat embeddings menggunakan model lokal...")
    # Menggunakan model embedding 'all-MiniLM-L6-v2'.
    # Model ini akan diunduh secara otomatis saat pertama kali dijalankan.
    # Proses ini berjalan di CPU Anda dan tidak memerlukan API key.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("Embeddings berhasil dibuat.")

    # 4. Simpan ke FAISS Vector Store
    print("Menyimpan ke Vector Store FAISS...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("Vector Store berhasil dibuat dan disimpan di 'vectorstore/db_faiss'")

if __name__ == "__main__":
    # Buat direktori jika belum ada
    if not os.path.exists("vectorstore"):
        os.makedirs("vectorstore")

    # Jalankan proses pembuatan vector store
    create_vector_store()