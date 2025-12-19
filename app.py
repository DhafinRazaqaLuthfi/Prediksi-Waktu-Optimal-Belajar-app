# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import os
from PIL import Image

# =====================
# KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# =====================
st.set_page_config(
    page_title="Prediksi Waktu Optimal Belajar",
    page_icon="📊",
    layout="centered"
)

# =====================
# HEADER ATAS (LOGO + INFO KELOMPOK)
# =====================
logo = Image.open("logo_itera.png")

col1, col2 = st.columns([1, 5])

with col1:
    st.image(logo, width=90)

with col2:
    st.markdown(
        """
        <div style="font-size:14px; color: gray;">
            <b>Mata Kuliah:</b> Deep Learning<br>
            <b>Kelompok 24</b> |
            Dhafin Razaqa Luthfi |
            Siti Nur Aarifah |
            Cyntia Kristina Sidauruk
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# =====================
# PATH FILE
# =====================
MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"

# =====================
# MODEL MLP
# =====================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.fc(x)

# =====================
# LOAD MODEL & SCALER
# =====================
if not os.path.exists(MODEL_PATH):
    st.error("❌ File model.pth tidak ditemukan")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ File scaler.pkl tidak ditemukan")
    st.stop()

model = MLP(input_dim=9)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

scaler = joblib.load(SCALER_PATH)

# =====================
# JUDUL APLIKASI
# =====================
st.title("📊 Prediksi Waktu Optimal Belajar")
st.write("Aplikasi prediksi waktu belajar optimal berbasis **MLP (PyTorch)**")

st.divider()
st.subheader("📝 Masukkan Data Kebiasaan Anda")

# =====================
# INPUT USER
# =====================
jam_tidur = st.selectbox(
    "Pada jam berapa Anda biasanya tidur?",
    ["21.00-22.00", "22.01-23.00", "23.01-00.00", "00.01-01.00", "> 01.00"]
)

durasi_tidur = st.selectbox(
    "Berapa lama Anda tidur per hari?",
    ["< 5 jam", "5-6 jam", "6-7 jam", "7-8 jam", "> 8 jam"]
)

durasi_belajar = st.selectbox(
    "Berapa lama Anda belajar per hari?",
    ["< 1 jam", "1-2 jam", "2-3 jam", "3-4 jam", "> 4 jam"]
)

jam_belajar = st.selectbox(
    "Biasanya mulai belajar jam berapa?",
    ["05.00-10.59", "11.00-16.59", "17.00-00.59"]
)

durasi_hp = st.selectbox(
    "Berapa lama Anda menggunakan HP/laptop selain belajar?",
    ["< 2 jam", "2-4 jam", "4-6 jam", "> 6 jam"]
)

coffee = st.selectbox(
    "Apakah sebelum belajar mengonsumsi kopi?",
    ["Tidak", "Ya"]
)

gangguan = st.slider("Berapa banyak gangguan saat belajar?", 1, 5)
mood = st.slider("Seberapa semangat Anda saat belajar?", 1, 5)
produktivitas = st.slider("Seberapa produktif Anda saat belajar?", 1, 5)

# =====================
# ENCODING
# =====================
encode_jam_tidur = {
    "21.00-22.00": 0,
    "22.01-23.00": 1,
    "23.01-00.00": 2,
    "00.01-01.00": 3,
    "> 01.00": 4
}

encode_durasi_tidur = {
    "< 5 jam": 0,
    "5-6 jam": 1,
    "6-7 jam": 2,
    "7-8 jam": 3,
    "> 8 jam": 4
}

encode_durasi_belajar = {
    "< 1 jam": 0,
    "1-2 jam": 1,
    "2-3 jam": 2,
    "3-4 jam": 3,
    "> 4 jam": 4
}

encode_jam_belajar = {
    "05.00-10.59": 0,
    "11.00-16.59": 1,
    "17.00-00.59": 2
}

encode_hp = {
    "< 2 jam": 0,
    "2-4 jam": 1,
    "4-6 jam": 2,
    "> 6 jam": 3
}

encode_coffee = {
    "Tidak": 0,
    "Ya": 1
}

# =====================
# PREDIKSI
# =====================
st.divider()

if st.button("🔮 Prediksi Waktu Optimal"):
    data = np.array([[
        encode_jam_tidur[jam_tidur],
        encode_durasi_tidur[durasi_tidur],
        encode_durasi_belajar[durasi_belajar],
        encode_jam_belajar[jam_belajar],
        gangguan,
        encode_hp[durasi_hp],
        encode_coffee[coffee],
        mood,
        produktivitas
    ]])

    data_scaled = scaler.transform(data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(data_tensor)
        probs = F.softmax(output, dim=1)[0].numpy()
        pred = np.argmax(probs)

    label_map = {
        0: "🌅 Pagi (05.00–10.59)",
        1: "🌤️ Siang (11.00–16.59)",
        2: "🌙 Malam (17.00–00.59)"
    }

    st.success(f"✅ **Waktu belajar paling optimal:** {label_map[pred]}")

    st.subheader("📊 Tingkat Keyakinan Model")

    labels = ["🌅 Pagi", "🌤️ Siang", "🌙 Malam"]
    for i, label in enumerate(labels):
        if i == pred:
            st.markdown(f"**{label} (PALING OPTIMAL)**")
        else:
            st.markdown(label)

        st.progress(float(probs[i]))
        st.write(f"{probs[i]*100:.2f}%")
