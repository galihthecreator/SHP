import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Model dan Scaler
@st.cache_resource # Memori cache agar tidak meload file berulang kali
def load_components():
    model = joblib.load('model_rf_crop.pkl')
    scaler = joblib.load('scaler_crop.pkl')
    return model, scaler

model, scaler = load_components()

# Konfigurasi Halaman Dasar
st.set_page_config(page_title="Rekomendasi Komoditas", layout="wide")

# Kustomisasi CSS untuk Tema Hijau-Cokelat
st.markdown("""
    <style>
    .stApp {
        background-color: #F9FBE7;
    }
    .css-1d391kg {  /* Sidebar */
        background-color: #5D4037;
    }
    h1, h2, h3 {
        color: #2E7D32 !important;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #5D4037;
    }
    /* Warna Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #2E7D32; 
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR: INPUT PARAMETER LAHAN
# ==========================================
st.sidebar.markdown("<h2 style='color: white;'>Parameter Lahan</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<h4 style='color: #A5D6A7;'>Unsur Hara Makro</h4>", unsafe_allow_html=True)
n_val = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
p_val = st.sidebar.slider("Phosphorus (P)", 0, 145, 50)
k_val = st.sidebar.slider("Potassium (K)", 0, 205, 50)

st.sidebar.markdown("<h4 style='color: #A5D6A7;'>Agro-Klimat</h4>", unsafe_allow_html=True)
temp_val = st.sidebar.slider("Suhu (°C)", 5.0, 50.0, 25.0)
hum_val = st.sidebar.slider("Kelembapan (%)", 10.0, 100.0, 70.0)
ph_val = st.sidebar.slider("pH Tanah", 3.5, 9.9, 6.5)
rain_val = st.sidebar.slider("Curah Hujan (mm)", 20.0, 300.0, 150.0)

analyze_btn = st.sidebar.button("Analisis Kesesuaian Lahan")

# ==========================================
# MAIN CANVAS: HASIL & ANALISIS
# ==========================================
st.title("Sistem Rekomendasi Komoditas Pertanian")
# Hapus st.title() dan st.markdown() bawaan, ganti dengan injeksi HTML Absolut
st.markdown(
    """
    <p style='color: #271C19; font-size: 1.1rem; line-height: 1.6; margin-top: 10px;'>
        Masukkan parameter bio-fisik lahan pada panel di sebelah kiri untuk mendapatkan rekomendasi komoditas berbasis algoritma klasifikasi <em>Ensemble Learning</em>.
    </p>
    """, 
    unsafe_allow_html=True
)

st.markdown("---")

if analyze_btn:
    # MOCKUP: Simulasi hasil model predict_proba()
    # Di dunia nyata, kalian harus meload model Random Forest kalian di sini (model.predict_proba(input))
    
    st.subheader("Hasil Rekomendasi Klasifikasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div style='background-color:#E8F5E9; padding:20px; border-radius:10px; border-left: 5px solid #2E7D32;'>"
                    "<h3 style='margin:0;'>1. Rice (Padi)</h3><p style='color:#5D4037; font-size:24px; font-weight:bold;'>88.5%</p></div>", 
                    unsafe_allow_html=True)
        st.progress(88)
        
    with col2:
        st.markdown("<div style='background-color:#FFF3E0; padding:20px; border-radius:10px; border-left: 5px solid #5D4037;'>"
                    "<h3 style='margin:0; color:#5D4037;'>2. Jute</h3><p style='color:#5D4037; font-size:24px; font-weight:bold;'>9.2%</p></div>", 
                    unsafe_allow_html=True)
        st.progress(9)
        
    with col3:
        st.markdown("<div style='background-color:#FFF3E0; padding:20px; border-radius:10px; border-left: 5px solid #5D4037;'>"
                    "<h3 style='margin:0; color:#5D4037;'>3. Papaya</h3><p style='color:#5D4037; font-size:24px; font-weight:bold;'>2.3%</p></div>", 
                    unsafe_allow_html=True)
        st.progress(2)

    st.markdown("---")
    st.subheader("Analisis Argumen Algoritma")
    st.info("Berdasarkan input, algoritma memberikan bobot dominan pada rekomendasi **Rice** karena indikator Curah Hujan ({} mm) dan Kelembapan ({} %) memenuhi *threshold* tinggi yang diperlukan varietas tersebut.".format(rain_val, hum_val))