# File: api_client.py

import requests
import streamlit as st

def get_answer_from_api(question: str, context: str, api_url: str):
    """
    Mengirim pertanyaan dan konteks ke API model yang dipilih.

    Args:
        question (str): Pertanyaan mentah dari pengguna.
        context (str): String konteks yang sudah diformat dari RAG.
        api_url (str): URL endpoint dari model yang dipilih.

    Returns:
        str: Jawaban dari model atau pesan error.
    """
    if not api_url:
        st.error("URL API untuk model yang dipilih tidak valid atau tidak tersedia.")
        return "Error: Konfigurasi API tidak ditemukan."

    # Asumsi endpoint predict selalu '/predict'
    full_url = f"{api_url.rstrip('/')}/predict"
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
        error_message = f"Koneksi Gagal: Tidak dapat menghubungi API di {full_url}. Detail: {e}"
        st.error(error_message)
        return error_message
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses respons: {e}")
        # return "Terjadi kesalahan pada sisi klien."