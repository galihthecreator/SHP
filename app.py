import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==========================================
# FASE 1: LOAD KOMPONEN AI (CACHED)
# ==========================================
@st.cache_resource
def load_ai_components():
    # Pastikan nama file ini sama persis dengan yang ada di foldermu/GitHub
    loaded_model = joblib.load('model_rf_crop.pkl')
    loaded_scaler = joblib.load('scaler_crop.pkl')
    return loaded_model, loaded_scaler

model, scaler = load_ai_components()

# ==========================================
# FASE 2: KONFIGURASI TEMA & UI
# ==========================================
st.set_page_config(page_title="Rekomendasi Komoditas", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F9FBE7; }
    .css-1d391kg { background-color: #5D4037; }
    .stButton>button {
        background-color: #2E7D32; color: white; border-radius: 8px; width: 100%; font-weight: bold;
    }
    .stButton>button:hover { background-color: #5D4037; }
    .stProgress > div > div > div > div { background-color: #2E7D32; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# FASE 3: PANEL INPUT (SIDEBAR)
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
# FASE 4: MAIN CANVAS & EKSEKUSI PREDIKSI
# ==========================================
st.markdown(
    """
    <h1 style='color: #1A1A1A; font-weight: 800; margin-bottom: 0px;'>Sistem Rekomendasi Komoditas Pertanian</h1>
    <p style='color: #271C19; font-size: 1.1rem; line-height: 1.6; margin-top: 10px;'>
        Masukkan parameter bio-fisik lahan pada panel di sebelah kiri untuk mendapatkan rekomendasi komoditas berbasis algoritma klasifikasi <em>Ensemble Learning</em>.
    </p>
    """, unsafe_allow_html=True
)
st.markdown("---")

if analyze_btn:
    # 1. Agregasi input dari slider menjadi matriks 2D (Sesuai urutan fitur saat training)
    input_data = np.array([[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]])
    
    # 2. Transformasi skala menggunakan Scaler historis
    input_scaled = scaler.transform(input_data)
    
    # 3. Eksekusi probabilitas model Random Forest
    probabilitas = model.predict_proba(input_scaled)[0]
    kelas_tanaman = model.classes_
    
    # 4. Ekstraksi Top 3 Prediksi
    top_3_idx = np.argsort(probabilitas)[-3:][::-1]
    
    c1, p1 = kelas_tanaman[top_3_idx[0]].upper(), probabilitas[top_3_idx[0]] * 100
    c2, p2 = kelas_tanaman[top_3_idx[1]].capitalize(), probabilitas[top_3_idx[1]] * 100
    c3, p3 = kelas_tanaman[top_3_idx[2]].capitalize(), probabilitas[top_3_idx[2]] * 100
    
    # 5. Injeksi variabel dinamis ke dalam UI menggunakan f-string (format string)
    st.subheader("Hasil Rekomendasi Klasifikasi")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<div style='background-color:#E8F5E9; padding:20px; border-radius:10px; border-left: 5px solid #2E7D32;'>"
                    f"<h3 style='margin:0; color:#1A1A1A;'>1. {c1}</h3><p style='color:#5D4037; font-size:24px; font-weight:bold;'>{p1:.1f}%</p></div>", 
                    unsafe_allow_html=True)
        st.progress(int(p1))
        
    with col2:
        st.markdown(f"<div style='background-color:#FFF3E0; padding:20px; border-radius:10px; border-left: 5px solid #5D4037;'>"
                    f"<h3 style='margin:0; color:#5D4037;'>2. {c2}</h3><p style='color:#5D4037; font-size:24px; font-weight:bold;'>{p2:.1f}%</p></div>", 
                    unsafe_allow_html=True)
        st.progress(int(p2))
        
    with col3:
        st.markdown(f"<div style='background-color:#FFF3E0; padding:20px; border-radius:10px; border-left: 5px solid #5D4037;'>"
                    f"<h3 style='margin:0; color:#5D4037;'>3. {c3}</h3><p style='color:#5D4037; font-size:24px; font-weight:bold;'>{p3:.1f}%</p></div>", 
                    unsafe_allow_html=True)
        st.progress(int(p3))

    st.markdown("---")
    st.subheader("Logika Inferensi Model")
    st.info(f"Berdasarkan probabilitas {p1:.1f}%, algoritma ensemble menetapkan **{c1}** sebagai kecocokan tertinggi untuk profil lahan dengan Curah Hujan {rain_val} mm, Kelembapan {hum_val}%, dan keseimbangan hara NPK terdeteksi.")