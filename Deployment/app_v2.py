# File: app.py (Dimodifikasi untuk RunPod Custom API)

import os
import time
import requests # <-- Tambahkan import ini
import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM # <-- Import dasar untuk LLM kustom
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, List, Any, Mapping

# Muat environment variable dari file .env
load_dotenv()

# --- KELAS LLM KUSTOM UNTUK RUNPOD ---
class RunPodLLM(LLM):
    """Kelas LLM kustom untuk berinteraksi dengan RunPod Serverless API."""
    endpoint_url: str
    api_key: str

    @property
    def _llm_type(self) -> str:
        return "runpod_custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Sesuai dengan struktur request di screenshot Anda
        payload = {
            "input": {
                "prompt": prompt,
                # Anda bisa menambahkan parameter lain di sini jika template RunPod Anda mendukungnya
                # "max_new_tokens": 512,
            }
        }

        # 1. Kirim request untuk memulai pekerjaan
        try:
            response = requests.post(f"{self.endpoint_url}/run", headers=headers, json=payload)
            response.raise_for_status()
            job = response.json()
            job_id = job.get("id")
            if not job_id:
                raise ValueError("Gagal mendapatkan job ID dari RunPod.")
        except requests.exceptions.RequestException as e:
            return f"Error saat memulai pekerjaan di RunPod: {e}"

        # 2. Polling status pekerjaan sampai selesai
        status_url = f"{self.endpoint_url}/status/{job_id}"
        timeout = time.time() + 300  # Timeout 5 menit
        
        while time.time() < timeout:
            try:
                status_response = requests.get(status_url, headers=headers)
                status_response.raise_for_status()
                status_data = status_response.json()

                if status_data.get("status") == "COMPLETED":
                    # 3. Ekstrak output sesuai struktur di screenshot
                    try:
                        # Path parsing sesuai screenshot: output -> choices[0] -> tokens[0] -> text
                        output_text = status_data['output']['choices'][0]['tokens'][0]['text']
                        return output_text
                    except (KeyError, IndexError, TypeError) as e:
                        return f"Error saat mem-parsing output dari RunPod: {e}. Respon penuh: {status_data}"

                elif status_data.get("status") in ["FAILED", "CANCELLED"]:
                    return f"Pekerjaan di RunPod gagal atau dibatalkan. Status: {status_data}"

                # Tunggu sebelum polling lagi
                time.sleep(2)

            except requests.exceptions.RequestException as e:
                return f"Error saat memeriksa status pekerjaan di RunPod: {e}"

        return "Error: Waktu tunggu untuk respon dari RunPod habis."
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint_url": self.endpoint_url}

# --- AKHIR KELAS KUSTOM ---

# Path ke vector store yang sudah ada
DB_FAISS_PATH = "vectorstore/db_faiss"

# Prompt template tetap sama
custom_prompt_template = """
### Peran:
Anda adalah seorang Ahli Hukum Strategis Indonesia yang sangat teliti dan objektif...
(Isi prompt Anda di sini, saya singkat agar tidak terlalu panjang)
...
### Konteks:
{context}

### Pertanyaan:
{question}

### Jawaban:
"""

def set_custom_prompt():
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    return prompt

def load_llm():
    """
    Memuat model LLM dari endpoint RunPod kustom.
    """
    endpoint = os.getenv("RUNPOD_API_ENDPOINT")
    api_key = os.getenv("RUNPOD_API_KEY")

    if not endpoint or not api_key:
        st.error("Error: RUNPOD_API_ENDPOINT atau RUNPOD_API_KEY tidak ditemukan. Pastikan file .env Anda sudah benar.")
        st.stop()

    # Menggunakan kelas LLM kustom kita
    llm = RunPodLLM(endpoint_url=endpoint, api_key=api_key)
    return llm

def create_rag_chain(retriever, llm, prompt):
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm # Langsung menggunakan objek LLM kita
        # StrOutputParser tidak diperlukan jika output LLM sudah string
    )
    return rag_chain

# --- UI Streamlit ---
st.title("⚖️ LawBot UU TNI No. 34 Tahun 2004")
st.write("Ajukan pertanyaan spesifik mengenai isi UU No. 34 Tahun 2004.")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 4})

    llm = load_llm()
    prompt = set_custom_prompt()
    qa_chain = create_rag_chain(retriever, llm, prompt)

    user_question = st.text_input("Ajukan pertanyaan Anda:", placeholder="Contoh: Apa saja tugas TNI dalam Operasi Militer Selain Perang?")

    if user_question:
        st.markdown("---")
        with st.spinner("Menghubungi model di RunPod dan menyusun jawaban..."):
            # Karena _call() sudah mengembalikan string, StrOutputParser tidak wajib
            # LangChain cukup pintar untuk menanganinya.
            response = qa_chain.invoke(user_question)
            st.markdown(response)

            with st.expander("Lihat Konteks Dokumen yang Digunakan untuk Analisis"):
                relevant_docs = retriever.invoke(user_question)
                for i, doc in enumerate(relevant_docs):
                    st.write(f"**Konteks {i+1} (Sumber: {doc.metadata.get('source', 'N/A')}, Hal: {doc.metadata.get('page', 'N/A')})**")
                    st.caption(doc.page_content)
                    st.write("---")

except FileNotFoundError:
    st.error("Vector Store tidak ditemukan. Harap jalankan `python setup_vectorstore.py` terlebih dahulu.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat menjalankan aplikasi: {e}")