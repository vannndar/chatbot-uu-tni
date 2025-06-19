import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Muat environment variable dari file .env
load_dotenv()

# Path ke vector store yang sudah ada
DB_FAISS_PATH = "vectorstore/db_faiss"

# Template prompt Anda sudah sangat baik, tidak perlu diubah.
custom_prompt_template = """
### Peran:
Anda adalah seorang Ahli Hukum Strategis Indonesia yang sangat teliti dan objektif. Tugas Anda adalah menganalisis pertanyaan berdasarkan konteks hukum yang disediakan dari Undang-Undang.

### Struktur Jawaban:
1.  **Inti Jawaban:**
    -   Berikan jawaban langsung dan ringkas terhadap pertanyaan pengguna.
    -   Sebutkan satu atau dua implikasi praktis utama dari jawaban tersebut.
2.  **Rincian Analisis:**
    a.  **Isu Pokok:** Identifikasi dan nyatakan kembali pertanyaan hukum spesifik yang diajukan.
    b.  **Aturan & Unsur:** Uraikan aturan hukum yang relevan dari konteks yang diberikan. **Wajib menyertakan kutipan langsung dan menyebutkan nomor Pasal dan Ayat** yang menjadi dasar analisis.
    c.  **Penerapan:** Analisis secara logis bagaimana aturan hukum tersebut berlaku untuk menjawab isu pokok.
3.  **Disclaimer:**
    -   Sertakan disclaimer bahwa jawaban ini bersifat informasional berdasarkan dokumen yang diberikan dan bukan merupakan nasihat hukum yang mengikat.

### Konteks:
{context}

### Pertanyaan:
{question}

### Jawaban:
"""

def load_llm():
    """Memuat model LLM dari Google Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Error: GOOGLE_API_KEY tidak ditemukan. Pastikan file .env Anda sudah benar.")
        st.stop()
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True
    )

# --- PERUBAHAN UTAMA DI FUNGSI INI ---
def create_rag_chain(retriever, llm, prompt):
    """
    Membuat RAG chain yang mengembalikan dictionary berisi jawaban dan konteks.
    Ini menghindari pencarian ganda.
    """
    def format_docs(docs):
        # Menggabungkan konten dokumen menjadi satu string untuk dimasukkan ke prompt
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        # Langkah 1: Ambil pertanyaan dan ambil dokumen yang relevan.
        # Kita menggunakan RunnablePassthrough untuk membawa dokumen asli ke langkah akhir.
        RunnablePassthrough.assign(
            context=(RunnableLambda(lambda x: x['question']) | retriever)
        )
        # Langkah 2: Buat dictionary baru untuk jawaban.
        # 'answer' akan berisi hasil dari LLM.
        # 'context' akan berisi dokumen yang kita bawa dari langkah sebelumnya.
        | RunnablePassthrough.assign(
            answer=(
                RunnablePassthrough.assign(
                    context=(lambda x: format_docs(x['context']))
                )
                | prompt
                | llm
                | StrOutputParser()
            )
        )
    )
    return rag_chain

# --- UI Streamlit ---
st.title("⚖️ LawBot UU TNI No. 34 Tahun 2004")
st.write("Ajukan pertanyaan spesifik mengenai isi UU No. 34 Tahun 2004 tentang Tentara Nasional Indonesia.")

try:
    # Model embedding yang digunakan harus SAMA PERSIS dengan di setup_vectorstore.py
    # Ganti 'paraphrase-multilingual-MiniLM-L12-v2' jika Anda menggunakan model lain saat setup
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 4})

    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    qa_chain = create_rag_chain(retriever, llm, prompt)

    user_question = st.text_input("Ajukan pertanyaan Anda:", placeholder="Contoh: Apa saja tugas TNI dalam Operasi Militer Selain Perang?")

    if user_question:
        st.markdown("---")
        with st.spinner("Menganalisis dokumen dan menyusun jawaban..."):
            # --- PERUBAHAN PADA CARA MEMANGGIL CHAIN ---
            # Input sekarang harus berupa dictionary agar chain tahu mana 'question'
            input_dict = {"question": user_question}
            
            # Melakukan pencarian SATU KALI SAJA
            result = qa_chain.invoke(input_dict)
            
            # Menampilkan jawaban dari dictionary hasil
            st.markdown(result['answer'])

            # Menampilkan konteks dari dictionary hasil
            with st.expander("Lihat Konteks Dokumen yang Digunakan untuk Analisis"):
                for i, doc in enumerate(result['context']):
                    st.write(f"**Konteks {i+1} (Sumber: {doc.metadata.get('source', 'N/A')}, Hal: {doc.metadata.get('page', 'N/A')})**")
                    st.caption(doc.page_content)
                    st.write("---")

except FileNotFoundError:
    st.error("Vector Store tidak ditemukan. Harap jalankan `python setup_vectorstore.py` terlebih dahulu.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat menjalankan aplikasi: {e}")