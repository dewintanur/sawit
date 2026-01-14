import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# =====================
# Konfigurasi Halaman
# =====================
st.set_page_config(
    page_title="Deteksi Sawit AI",
    page_icon="🌴",
    layout="wide"
)

# =====================
# Header
# =====================
st.markdown(
    """
    <h1 style='text-align: center;'>🌴 Deteksi Kematangan Sawit Berbasis AI</h1>
    <p style='text-align: center;'>
    Upload gambar atau gunakan kamera untuk mendeteksi tingkat kematangan buah sawit secara otomatis
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =====================
# Load Model
# =====================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    model_status = "✅ Model YOLO berhasil dimuat"
except Exception:
    model = None
    model_status = "❌ Model tidak ditemukan (best.pt)"

# =====================
# Sidebar
# =====================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.info(model_status)

    input_mode = st.radio(
        "Pilih Metode Input:",
        ["📷 Kamera", "📁 Upload Gambar"]
    )

    st.markdown("---")
    st.caption("Dikembangkan menggunakan YOLO & Streamlit")

# =====================
# Input Gambar
# =====================
img_file = None

if input_mode == "📷 Kamera":
    img_file = st.camera_input("Ambil Foto Sawit")
else:
    img_file = st.file_uploader(
        "Upload Gambar Sawit",
        type=["jpg", "jpeg", "png"]
    )

# =====================
# Proses Deteksi
# =====================
if img_file is not None and model is not None:
    image = Image.open(img_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Gambar Asli")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 Hasil Deteksi AI")
        results = model(image)

        res_plotted = results[0].plot()
        st.image(res_plotted, use_container_width=True)

    # =====================
    # Ringkasan Deteksi
    # =====================
    st.markdown("## 📊 Ringkasan Hasil Deteksi")

    detected_classes = []
    names = model.names

    for r in results:
        for c in r.boxes.cls:
            detected_classes.append(names[int(c)])

    if detected_classes:
        count = Counter(detected_classes)

        for label, jumlah in count.items():
            st.success(f"**{label}** : {jumlah} buah terdeteksi")

    else:
        st.warning("Tidak ada buah sawit terdeteksi")

# =====================
# Footer
# =====================
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:12px;'>
    Sistem Deteksi Kematangan Sawit | YOLOv8 | Streamlit Cloud
    </p>
    """,
    unsafe_allow_html=True
)
