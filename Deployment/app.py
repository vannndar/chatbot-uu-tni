# File: app.py

import streamlit as st
from rag_handler import load_retriever, get_context
from api_client import get_answer_from_api
# Modifikasi Impor: Impor fungsi, bukan variabel statis
from config import load_and_get_api_models
import os

# Path aset tetap sama
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo_its.png")
USER_AVATAR_PATH = os.path.join(os.path.dirname(__file__), "assets", "user_icon.png")
BOT_AVATAR_PATH = os.path.join(os.path.dirname(__file__), "assets", "law_icon.png")

def display_sources(documents):
    with st.expander("Lihat Konteks Dokumen yang Digunakan untuk Analisis"):
        if documents:
            for i, doc in enumerate(documents):
                source_text = f"""
                **Konteks {i+1}:**
                > {doc.page_content}
                """
                st.markdown(source_text, unsafe_allow_html=True)
                st.divider()
        else:
            st.write("Tidak ada dokumen relevan yang ditemukan untuk pertanyaan ini.")

# Modifikasi: Fungsi sidebar sekarang menerima konfigurasi model sebagai argumen
def setup_sidebar(api_models):
    with st.sidebar:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(LOGO_PATH, width=50)
        with col2:
            st.markdown("<h4 style='margin-top: 15px;'>Text Mining Project</h4>", unsafe_allow_html=True)
        
        st.divider()
        st.header("Pengaturan Model")
        
        # Ambil daftar nama model dari dictionary yang diterima
        model_names = list(api_models.keys())
        
        # Buat selectbox untuk memilih model
        selected_model = st.selectbox(
            "Pilih Model AI:",
            options=model_names,
            index=0, # Default ke model pertama
            key="selected_model_name"
        )

        st.header("Tentang LawBot")
        st.info(
            "LawBot ini dirancang untuk menjawab pertanyaan spesifik mengenai "
            "UU TNI No. 34 Tahun 2004."
        )
        
        st.divider()
        st.header("Contoh Pertanyaan")
        
        example_questions = [
            "Apakah TNI bisa menjabat sebagai walikota?",
            "Apa saja tugas TNI dalam operasi militer selain perang?",
            "Bagaimana proses pengangkatan Panglima TNI?"
        ]
        
        clicked_question = None
        for q in example_questions:
            if st.button(q, use_container_width=True):
                clicked_question = q
        
        st.divider()

        if st.button("Bersihkan Riwayat Chat", use_container_width=True, type="primary"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.markdown(
        """
        **Dikembangkan oleh:**
        * Thariq Ivan - 5025221013
        * Lucky Santoso - 5025221050
        * Naufal Khairul R - 5025221127
        """
        )
            
        return clicked_question

def main():
    st.set_page_config(page_title="LawBot UU TNI", page_icon=LOGO_PATH, layout="wide")

    # --- MODIFIKASI KUNCI ---
    # Panggil fungsi untuk memuat konfigurasi API setiap kali script dijalankan.
    # Ini memastikan perubahan di .env akan langsung terlihat.
    API_MODELS = load_and_get_api_models()

    # Cek jika ada model yang terkonfigurasi setelah memuat
    if not API_MODELS:
        st.error("Konfigurasi Error: Tidak ada model API (dengan format API_URL_NAMAMODEL) yang ditemukan di file .env. Mohon periksa konfigurasi Anda.")
        st.stop()

    retriever = load_retriever()
    if retriever is None:
        st.error("Gagal memuat komponen RAG. Aplikasi tidak dapat berjalan.")
        st.stop()
    
    # Modifikasi: Kirim dictionary model yang sudah dimuat ke sidebar
    example_question = setup_sidebar(API_MODELS)
    
    # Ambil nama model yang sedang dipilih dari session_state
    current_model_name = st.session_state.get("selected_model_name")

    # Penanganan jika model yang dipilih tidak lagi ada di .env setelah update
    if not current_model_name or current_model_name not in API_MODELS:
        st.warning("Model yang sebelumnya dipilih tidak lagi tersedia. Menggunakan model default.")
        # Reset ke model pertama yang tersedia
        current_model_name = list(API_MODELS.keys())[0]
        st.session_state.selected_model_name = current_model_name
        st.rerun() # Lakukan rerun agar UI update dengan benar
    
    st.title("LawBot UU TNI No. 34 Tahun 2004")
    st.caption(f"Didukung oleh Model: **{current_model_name}** & RAG")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = USER_AVATAR_PATH if message["role"] == "user" else BOT_AVATAR_PATH
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "model_used" in message:
                st.caption(f"Dijawab menggunakan: {message['model_used']}")
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])

    user_question = st.chat_input("Ajukan pertanyaan Anda di sini...")
    if example_question:
        user_question = example_question

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user", avatar=USER_AVATAR_PATH):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar=BOT_AVATAR_PATH):
            with st.spinner(f"Menghubungi model {current_model_name}..."):

                context_text, relevant_docs = get_context(retriever, user_question)
      
                selected_api_url = API_MODELS.get(current_model_name)

                answer = get_answer_from_api(user_question, context_text, selected_api_url)
                
                st.markdown(answer)
                st.caption(f"Dijawab menggunakan: {current_model_name}")
                display_sources(relevant_docs)
   
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": relevant_docs,
            "model_used": current_model_name # Simpan nama model
        })
        
        if example_question:
            st.rerun() 

if __name__ == "__main__":
    main()