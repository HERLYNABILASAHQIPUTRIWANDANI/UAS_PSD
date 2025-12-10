# app.py
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Time Series Classification (SVM)", layout="wide")

# Load model & scaler
@st.cache_resource
def load_svm_model():
    with open("models/svm_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("models/scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_svm_model()
scaler = load_scaler()

# Mapping kelas
class_names = {
    1: {"name": "Normal", "description": "Stabil, tidak naik atau turun"},
    2: {"name": "Cyclic", "description": "Bergelombang, naik-turun berulang"},
    3: {"name": "Increasing trend", "description": "Tren naik secara bertahap"},
    4: {"name": "Decreasing trend", "description": "Tren turun secara bertahap"},
    5: {"name": "Upward shift", "description": "Naik tiba-tiba, tetap tinggi"},
    6: {"name": "Downward shift", "description": "Turun tiba-tiba, tetap rendah"},
    7: {"name": "Valley", "description": "Turun lalu naik, membentuk lembah (U-shape)"}
}

# Fungsi baca TXT yang fleksibel
def parse_text_series(raw):
    # Ubah semua pemisah ke koma
    separators = ["\r\n", "\n", "\t", "  ", " "]
    for sep in separators:
        raw = raw.replace(sep, ",")

    # Split dan ambil angka valid
    values = [v.strip() for v in raw.split(",") if v.strip() != ""]

    try:
        return np.array([float(v) for v in values])
    except:
        return None


# UI
st.title("📈 Time Series Classification – Synthetic Control (SVM)")
st.write("Upload file .txt atau input manual deretan 60 angka time-series")

manual_text = st.text_area("Input manual (pisahkan koma atau enter)", height=120)
uploaded_file = st.file_uploader("Upload file (.txt)", type=["txt"])
predict_button = st.button("🔮 Predict")

if predict_button:
    data = None

    # Input manual
    if manual_text.strip() != "":
        raw = manual_text
        data = parse_text_series(raw)
        if data is None:
            st.error("Format input manual salah!")
            st.stop()

    # Input dari file
    elif uploaded_file:
        try:
            raw = uploaded_file.read().decode("utf-8")
            data = parse_text_series(raw)
        except:
            st.error("File tidak dapat dibaca!")
            st.stop()

    if data is None:
        st.error("Tidak ada data!")
        st.stop()

    if len(data) != 60:
        st.error(f"❌ Jumlah nilai harus 60, tapi ditemukan {len(data)} angka!")
        st.stop()

    data_2d = data.reshape(1, -1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data, marker='o', linewidth=2)
    ax.set_title("Time Series Input")
    ax.grid(True)
    st.pyplot(fig)

    # Scaling + Prediksi
    data_scaled = scaler.transform(data_2d)
    prediction = int(model.predict(data_scaled)[0])

    # Confidence
    try:
        confidence = float(np.max(model.predict_proba(data_scaled)[0]))
        confidence_type = "Probability (SVM)"
    except:
        scores = model.decision_function(data_scaled)
        confidence = float(np.max(scores) / np.sum(np.abs(scores)))
        confidence_type = "Decision Function"

    pred_info = class_names.get(prediction, {"name": "Unknown", "description": "Unknown"})
    st.success(f"📌 Predicted Class: {prediction} – {pred_info['name']}")
    st.write(f"**Deskripsi pola:** {pred_info['description']}")
    st.info(f"🔥 Confidence ({confidence_type}): {confidence*100:.2f}%")

else:
    st.info("Klik tombol Predict setelah input data")
