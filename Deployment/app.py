# File: app.py

import streamlit as st
from rag_handler import load_retriever, get_context
from api_client import get_answer_from_api

LOGO_PATH = "assets/logo_its.png"
USER_AVATAR_PATH = "assets/user_icon.png"
BOT_AVATAR_PATH = "assets/law_icon.png"

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

def setup_sidebar():
    with st.sidebar:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(LOGO_PATH, width=50)
        with col2:
            st.markdown("<h4 style='margin-top: 15px;'>Text Mining Project</h4>", unsafe_allow_html=True)
        
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

    retriever = load_retriever()
    if retriever is None:
        st.error("Gagal memuat komponen RAG. Aplikasi tidak dapat berjalan.")
        st.stop()
    
    example_question = setup_sidebar()
    
    st.title("LawBot UU TNI No. 34 Tahun 2004")
    st.caption("Didukung oleh Model AI Fine-tuned & RAG")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = USER_AVATAR_PATH if message["role"] == "user" else BOT_AVATAR_PATH
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])

    user_question = st.chat_input("Ajukan pertanyaan Anda di sini...")
    if example_question:
        user_question = example_question

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question, "sources": []})
        with st.chat_message("user", avatar=USER_AVATAR_PATH):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar=BOT_AVATAR_PATH):
            with st.spinner("Memproses pertanyaan..."):
                context_text, relevant_docs = get_context(retriever, user_question)
                answer = get_answer_from_api(user_question, context_text)
                
                st.markdown(answer)
                display_sources(relevant_docs)
        
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": relevant_docs})
        
        if example_question:
            st.stop()

if __name__ == "__main__":
    main()
