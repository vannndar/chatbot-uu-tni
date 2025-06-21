# File: rag_handler.py

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import VECTORSTORE_PATH, EMBEDDING_MODEL_NAME, RETRIEVER_SEARCH_K

def format_docs(docs):
    """Mengubah list objek Document dari FAISS menjadi satu string teks."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def load_retriever():
    """
    Memuat embeddings dan vector store, lalu mengembalikannya sebagai retriever.
    Menggunakan cache Streamlit untuk performa.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return db.as_retriever(search_kwargs={'k': RETRIEVER_SEARCH_K})
    except FileNotFoundError:
        st.error(f"Vector Store tidak ditemukan di path: '{VECTORSTORE_PATH}'. Pastikan path sudah benar.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat komponen RAG: {e}")
        return None

def get_context(retriever, question: str):
    """
    Mengambil dokumen relevan dan memformatnya menjadi string konteks.
    
    Returns:
        tuple: (string konteks, list dokumen asli)
    """
    if retriever is None:
        return "Konteks tidak tersedia karena retriever gagal dimuat.", []
        
    relevant_docs = retriever.invoke(question)
    context_text = format_docs(relevant_docs)
    return context_text, relevant_docs