import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Deteksi Sawit Cloud", page_icon="🌴")
st.title("🌴 Deteksi Kematangan Sawit (Cloud)")

# Load Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Model tidak ditemukan. Pastikan best.pt sudah diupload.")

# Input Kamera
img_file = st.camera_input("Jepret Foto Sawit")

if img_file is not None:
    # Proses
    image = Image.open(img_file)
    results = model(image)
    
    # Tampilkan Hasil
    res_plotted = results[0].plot()[:, :, ::-1] # Konversi warna
    st.image(res_plotted, caption="Hasil Analisa AI", use_container_width=True)
    
    # Tampilkan Label (Opsional)
    names = model.names
    for r in results:
        for c in r.boxes.cls:
            st.info(f"Terdeteksi: {names[int(c)]}")