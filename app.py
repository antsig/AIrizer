import streamlit as st
import os
import sys
import json
import pandas as pd

# Menambahkan root directory ke path agar bisa import dari src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.summarizer import DocumentSummarizer
from src.core.classifier import DocumentClassifier
from src.core.keyword_extractor import KeywordExtractor
from src.core.action_extractor import ActionItemExtractor
from src.core.semantic_search import SemanticSearchEngine
from src.core.qa_engine import DocumentQA
from src.utils.document_parser import DocumentParser
from io import BytesIO
import glob

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Page Config ---
st.set_page_config(
    page_title="Document Intelligence AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');
    
    /* Apply premium typography globally without overriding icon fonts */
    html, body, .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Apply premium typography to input controls but preserve icons */
    input, button, select, textarea {
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Headers */
    .main-header {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 50%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.75;
        margin-bottom: 24px;
        line-height: 1.4;
    }
    
    /* Glassmorphism Cards with Smooth Transitions */
    .card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(128, 128, 128, 0.15);
        color: var(--text-color);
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(30, 136, 229, 0.1);
        border-color: rgba(30, 136, 229, 0.3);
    }
    
    /* Pill Badges with Hover Effect */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.08) 0%, rgba(21, 101, 192, 0.12) 100%);
        color: #1E88E5 !important;
        border: 1px solid rgba(30, 136, 229, 0.25);
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    
    .badge:hover {
        background: #1E88E5;
        color: white !important;
        transform: scale(1.05);
        box-shadow: 0 4px 10px rgba(30, 136, 229, 0.2);
    }
    
    /* Action Items Sleek Border & Hover Transform */
    .action-item {
        background: rgba(255, 152, 0, 0.03);
        color: var(--text-color);
        border-left: 5px solid #FF9800;
        border-top: 1px solid rgba(255, 152, 0, 0.12);
        border-right: 1px solid rgba(255, 152, 0, 0.12);
        border-bottom: 1px solid rgba(255, 152, 0, 0.12);
        padding: 14px 18px;
        margin-bottom: 10px;
        border-radius: 0 10px 10px 0;
        transition: all 0.2s ease;
    }
    
    .action-item:hover {
        transform: translateX(4px);
        background: rgba(255, 152, 0, 0.06);
    }
    
    /* Smooth Scrollbar for Premium Feel */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(128, 128, 128, 0.3);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(128, 128, 128, 0.5);
    }

    /* Mobile Responsiveness (Responsive Typography & Spacing) */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.9rem !important;
            line-height: 1.2;
            text-align: center;
        }
        .sub-header {
            font-size: 0.95rem !important;
            text-align: center;
            margin-bottom: 18px !important;
        }
        .card {
            padding: 16px !important;
            margin-bottom: 14px !important;
        }
        .badge {
            font-size: 0.8rem !important;
            padding: 4px 10px !important;
        }
        .action-item {
            padding: 10px 14px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Load Models (Cached for performance) ---
@st.cache_resource(show_spinner=False)
def load_models():
    return {
        "summarizer": DocumentSummarizer(),
        "classifier": DocumentClassifier(),
        "keyword_ext": KeywordExtractor(),
        "action_ext": ActionItemExtractor(),
        "search_engine": SemanticSearchEngine(),
        "qa_engine": DocumentQA()
    }

import time

# Placeholder for loading graphics
if 'models_loaded' not in st.session_state:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        # Pure CSS Loading Spinner (No internet required, instant load)
        css_spinner = """
        <style>
        .loader {
          border: 12px solid rgba(0, 0, 0, 0.1);
          border-left-color: #1E88E5;
          border-radius: 50%;
          width: 80px;
          height: 80px;
          animation: spin 1s linear infinite;
          margin: 0 auto 20px auto;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        </style>
        <div class="loader"></div>
        <h3 style='text-align: center; color: #1E88E5; font-family: sans-serif;'>🚀 Menginisialisasi AI Engine...</h3>
        """
        st.markdown(css_spinner, unsafe_allow_html=True)
        
        st.info("💡 **Tahukah Anda?** Sistem saat ini sedang memuat model *Natural Language Processing* (NLP) yang berukuran cukup besar ke dalam memori server/komputer Anda. Proses ini memakan waktu beberapa saat dan hanya terjadi pada saat pertama kali aplikasi dijalankan.")
        
    # Jeda sesaat agar browser sempat me-render elemen UI di atas sebelum thread diblokir oleh AI
    time.sleep(0.2)
    models = load_models()
    st.session_state['models_loaded'] = True
    loading_placeholder.empty() # Hapus tampilan loading setelah selesai
else:
    models = load_models()

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3214/3214746.png", width=100)
st.sidebar.title("Navigasi")
app_mode = st.sidebar.radio("Pilih Mode:", ["Analisis Dokumen", "Pencarian Semantik (Vector DB)"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tentang Sistem**\n\n"
    "Sistem AI Document Intelligence canggih yang menggabungkan model **mT5** (peringkasan), "
    "**IndoBERT-QA** (tanya-jawab), **mDeBERTa-v3** (klasifikasi zero-shot), "
    "**KeyBERT** (kata kunci), dan **ChromaDB** (pencarian semantik)."
)

# --- Mode 1: Document Analysis ---
if app_mode == "Analisis Dokumen":
    st.markdown('<div class="main-header">Analisis Dokumen Berbasis AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Otomatisasi Ringkasan, Klasifikasi, dan Ekstraksi Tugas</div>', unsafe_allow_html=True)

    # Input method
    input_method = st.radio("Metode Input:", ["Pilih dari Dataset Sintetik", "Ketik/Paste Teks Sendiri", "Upload File (PDF/DOCX/TXT)"])
    
    text_input = ""
    
    if input_method == "Pilih dari Dataset Sintetik":
        # Load synthetic data
        json_path = "data/synthetic/dataset.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc_titles = [d["title"] for d in data]
            selected_title = st.selectbox("Pilih Dokumen:", doc_titles)
            
            for d in data:
                if d["title"] == selected_title:
                    text_input = d["content"]
                    break
        else:
            st.warning("Dataset sintetik belum dibuat. Silakan jalankan `python src/utils/data_generator.py`")
            text_input = st.text_area("Masukkan teks dokumen di sini:", height=200)
            
    elif input_method == "Ketik/Paste Teks Sendiri":
        text_input = st.text_area("Masukkan teks dokumen di sini:", height=200)
        
    else: # Upload File
        uploaded_file = st.file_uploader("Unggah dokumen", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1]
            bytes_data = BytesIO(uploaded_file.getvalue())
            text_input = DocumentParser.extract_text(bytes_data, file_type=file_extension)
            if not text_input:
                st.error("Gagal mengekstrak teks dari dokumen atau dokumen kosong.")
            else:
                st.success(f"Berhasil mengekstrak {len(text_input)} karakter dari {uploaded_file.name}.")
                with st.expander("Lihat Teks yang Diekstrak"):
                    st.write(text_input[:1000] + ("..." if len(text_input) > 1000 else ""))

    if 'last_analyzed_text' not in st.session_state:
        st.session_state['last_analyzed_text'] = ""
        
    is_analyze_disabled = (text_input == st.session_state['last_analyzed_text']) or (not text_input)

    if st.button("Analisis Sekarang", type="primary", disabled=is_analyze_disabled) and text_input:
        st.session_state['last_analyzed_text'] = text_input
        with st.spinner("AI sedang menganalisis dokumen..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # --- Eksekusi Model NLP ---
            status_text.text("Tahap 1/4: Menghasilkan ringkasan dokumen...")
            progress_bar.progress(25)
            summary = models["summarizer"].summarize(text_input)
            
            status_text.text("Tahap 2/4: Mengekstrak action items...")
            progress_bar.progress(50)
            action_items = models["action_ext"].extract_action_items(text_input)
            
            status_text.text("Tahap 3/4: Mengklasifikasikan dokumen...")
            progress_bar.progress(75)
            # Mendapatkan label dan akurasi (score)
            classification_result = models["classifier"].classify(text_input, return_all_scores=True)
            if isinstance(classification_result, dict):
                label = classification_result["labels"][0]
                accuracy = classification_result["scores"][0]
            else:
                label = classification_result
                accuracy = None
                
            status_text.text("Tahap 4/4: Mengekstrak kata kunci...")
            progress_bar.progress(90)
            keywords = models["keyword_ext"].extract_keywords(text_input, top_n=5)
            
            progress_bar.progress(100)
            status_text.empty()
            time.sleep(0.2)
            progress_bar.empty()
            
            col1, col2 = st.columns([2, 1])
            
            # Kolom 1: Ringkasan & Action Items
            with col1:
                st.subheader("📝 Ringkasan Dokumen")
                st.markdown(f'<div class="card">{summary}</div>', unsafe_allow_html=True)
                
                # Logika Tampilan: Tampilkan hanya jika kategori masuk akal atau jika ada tugas yang terekstrak
                actionable_categories = ["Laporan", "Surat Resmi", "Invoice"]
                if label in actionable_categories or action_items:
                    st.subheader("⚡ Action Items (Tugas & Deadline)")
                    if action_items:
                        for item in action_items:
                            st.markdown(f'<div class="action-item">📌 {item}</div>', unsafe_allow_html=True)
                    else:
                        st.info("Tidak terdeteksi adanya penugasan atau deadline dalam dokumen ini.")
            
            # Kolom 2: Metadata (Klasifikasi & Keywords)
            with col2:
                st.subheader("🏷️ Klasifikasi")
                if accuracy is not None:
                    acc_percent = round(accuracy * 100, 2)
                    st.markdown(f'<div class="card" style="text-align:center;">'
                                f'<div style="font-size:1.2rem; font-weight:bold; color:#1E88E5;">{label}</div>'
                                f'<div style="font-size:0.9rem; color:#888; margin-top:5px;">Akurasi (Confidence): {acc_percent}%</div>'
                                f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="card" style="text-align:center; font-size:1.2rem; font-weight:bold; color:#1E88E5;">{label}</div>', unsafe_allow_html=True)
                
                st.subheader("🔑 Kata Kunci")
                kw_html = ""
                for kw in keywords:
                    kw_html += f'<span class="badge">{kw}</span>'
                st.markdown(f'<div class="card">{kw_html}</div>', unsafe_allow_html=True)
                
            # Tambahkan ke Vector DB di background agar bisa dicari nanti
            if input_method == "Ketik/Paste Teks Sendiri":
                doc_id = f"doc_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                models["search_engine"].add_documents([text_input], [{"source": "manual_input", "category": label}], [doc_id])

    # --- Fitur Chat dengan Dokumen ---
    if text_input and st.session_state.get('last_analyzed_text') == text_input:
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.subheader("💬 Chat dengan Dokumen (Q&A)")
        
        # Pilihan Mode QA
        chat_mode = st.radio("Metode AI:", ["Lokal (IndoBERT Extractive - Gratis)", "Cloud API (OpenAI Generative)"], horizontal=True)
        
        # Reset chat history jika dokumen berubah
        if st.session_state.get('chat_doc') != text_input:
            st.session_state['messages'] = []
            st.session_state['chat_doc'] = text_input
            
        # Tampilkan riwayat chat
        for msg in st.session_state.get('messages', []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Input Chat
        if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen di atas..."):
            # Tambahkan pertanyaan user ke state
            st.session_state['messages'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Proses jawaban AI
            with st.chat_message("assistant"):
                with st.spinner("AI sedang memikirkan jawaban..."):
                    if "Lokal" in chat_mode:
                        answer = models["qa_engine"].answer_question(prompt, text_input)
                    else:
                        api_key = os.getenv("OPENAI_API_KEY", "")
                        if not api_key:
                            answer = "⚠️ **OPENAI_API_KEY tidak ditemukan**. Silakan tambahkan kunci API Anda pada file `.env` di direktori proyek."
                        else:
                            answer = models["qa_engine"].answer_question_generative(prompt, text_input, api_key=api_key, provider="openai")
                    st.markdown(answer)
            # Simpan ke state
            st.session_state['messages'].append({"role": "assistant", "content": answer})

# --- Mode 2: Semantic Search ---
elif app_mode == "Pencarian Semantik (Vector DB)":
    st.markdown('<div class="main-header">Pencarian Semantik</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cari informasi berdasarkan makna dan konteks, bukan sekadar kecocokan kata.</div>', unsafe_allow_html=True)
    
    # Initialize DB with synthetic data if it's empty
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Index Ulang Dataset Sintetik"):
            json_path = "data/synthetic/dataset.json"
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                docs = [d["content"] for d in data]
                metadatas = [{"title": d["title"], "category": d["category"]} for d in data]
                ids = [d["id"] for d in data]
                
                with st.spinner("Memasukkan dokumen ke Vector Database..."):
                    models["search_engine"].add_documents(docs, metadatas, ids)
                st.success("Berhasil melakukan indexing dataset sintetik!")
            else:
                st.error("Dataset sintetik tidak ditemukan.")
    
    st.markdown("---")
    st.subheader("📁 Index dari Folder Custom")
    
    if 'last_indexed_path' not in st.session_state:
        st.session_state['last_indexed_path'] = ""
    if 'selected_folder_path' not in st.session_state:
        st.session_state['selected_folder_path'] = ""
        
    col_path1, col_path2 = st.columns([4, 1])
    with col_path1:
        custom_path = st.text_input("Path Folder Dokumen (contoh: C:\\Users\\Documents):", 
                                    value=st.session_state['selected_folder_path'],
                                    placeholder="Masukkan path folder absolut...")
    with col_path2:
        st.write("") # Spacer vertical alignment
        st.write("") # Spacer vertical alignment
        if st.button("📂 Browse...", use_container_width=True):
            try:
                import tkinter as tk
                from tkinter import filedialog
                
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)
                folder = filedialog.askdirectory(master=root)
                root.destroy()
                
                if folder:
                    # Normalisasi path untuk Windows
                    folder = os.path.normpath(folder)
                    st.session_state['selected_folder_path'] = folder
                    st.rerun()
            except Exception as e:
                st.error("Fitur 'Browse' tidak didukung di lingkungan cloud ini (headless server). Silakan masukkan path folder secara manual pada kolom input di samping.")

    # Sinkronisasi state jika pengguna mengetik manual
    if custom_path != st.session_state['selected_folder_path']:
        st.session_state['selected_folder_path'] = custom_path

    is_index_disabled = (custom_path == st.session_state['last_indexed_path']) or (not custom_path)
    
    if st.button("Index dari Folder", disabled=is_index_disabled):
        st.session_state['last_indexed_path'] = custom_path
        if os.path.exists(custom_path) and os.path.isdir(custom_path):
            with st.spinner(f"Membaca dokumen dari {custom_path}..."):
                docs = []
                metadatas = []
                ids = []
                
                files_to_index = []
                for ext in ["**/*.txt", "**/*.pdf", "**/*.docx"]:
                    files_to_index.extend(glob.glob(os.path.join(custom_path, ext), recursive=True))
                
                if not files_to_index:
                    st.warning("Tidak ditemukan dokumen teks/pdf/docx yang valid di folder tersebut.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file_path in enumerate(files_to_index):
                        filename = os.path.basename(file_path)
                        status_text.text(f"Memproses {filename} ({i+1}/{len(files_to_index)})...")
                        
                        file_ext = file_path.split('.')[-1]
                        text = DocumentParser.extract_text(file_path, file_type=file_ext)
                        if text and len(text.strip()) > 10:
                            docs.append(text)
                            metadatas.append({"title": filename, "source": "custom_folder", "category": "Custom"})
                            ids.append(f"custom_{filename}_{len(docs)}")
                            
                        # Update progress bar
                        progress_val = int(((i+1) / len(files_to_index)) * 100)
                        progress_bar.progress(progress_val)
                
                    status_text.empty()
                    progress_bar.empty()
                    
                    if docs:
                        models["search_engine"].add_documents(docs, metadatas, ids)
                        st.success(f"Berhasil melakukan indexing {len(docs)} dokumen dari folder!")
                    else:
                        st.warning("Gagal mengekstrak teks dari dokumen yang ditemukan.")
        else:
            st.error("Path folder tidak valid atau tidak ditemukan.")
            
    st.markdown("---")
    
    if 'last_search_query' not in st.session_state:
        st.session_state['last_search_query'] = ""
        
    st.write("🔍 Masukkan pertanyaan atau topik pencarian (misal: 'berapa anggaran tahun depan?'):")
    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        query = st.text_input("Query", placeholder="Ketik di sini...", label_visibility="collapsed")
    with col_q2:
        is_search_disabled = (query == st.session_state['last_search_query']) or (not query)
        search_clicked = st.button("Cari Dokumen", type="primary", use_container_width=True, disabled=is_search_disabled)
    
    if search_clicked and query:
        st.session_state['last_search_query'] = query
        with st.spinner("Mencari di Vector Database..."):
            progress_bar = st.progress(50)
            status_text = st.empty()
            status_text.text("Menghitung kedekatan semantik vektor...")
            
            results = models["search_engine"].search(query, top_k=3)
            
            progress_bar.progress(100)
            time.sleep(0.2)
            progress_bar.empty()
            status_text.empty()
            
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            if not docs:
                st.warning("Tidak ada dokumen yang ditemukan.")
            else:
                st.success(f"Ditemukan {len(docs)} hasil yang relevan.")
                
                for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
                    # Distance in Chroma is L2/Cosine distance. Lower is better.
                    similarity_score = round((1.0 - dist) * 100, 2)
                    
                    st.markdown(f"""
                    <div style="padding:15px; border:1px solid #ddd; border-radius:8px; margin-bottom:15px;">
                        <h4>{meta.get('title', 'Dokumen')} <span style="font-size:0.8em; font-weight:normal; color:#888;">(Similarity: {similarity_score}%)</span></h4>
                        <span class="badge" style="background-color:#4CAF50;">{meta.get('category', 'Unknown')}</span>
                        <p style="margin-top:10px;">{doc}</p>
                    </div>
                    """, unsafe_allow_html=True)
