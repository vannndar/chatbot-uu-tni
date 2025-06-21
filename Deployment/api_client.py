# File: api_client.py
import requests
import streamlit as st
from config import FLASK_API_URL  

def get_answer_from_api(question: str, context: str):
    """
    Mengirim pertanyaan dan konteks mentah ke Flask API.

    Args:
        question (str): Pertanyaan mentah dari pengguna.
        context (str): String konteks yang sudah diformat dari RAG.

    Returns:
        str: Jawaban dari model atau pesan error.
    """
    if not FLASK_API_URL:
        st.error("FLASK_API_URL tidak ditemukan di file .env atau config.py.")
        return "Error: Konfigurasi API tidak ditemukan."

    full_url = f"{FLASK_API_URL}/predict"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "question": question,
        "context": context
    }

    try:
        response = requests.post(full_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result.get("answer", "Error: Kunci 'answer' tidak ditemukan dalam respons API.")
    except requests.exceptions.RequestException as e:
        st.error(f"Koneksi Gagal: Tidak dapat menghubungi API. Detail: {e}")
        return "Gagal terhubung ke server model."
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses respons: {e}")
        return "Terjadi kesalahan pada sisi klien."