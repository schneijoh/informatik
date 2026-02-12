import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import json
import os
from datetime import datetime

st.set_page_config(page_title="KI Fundbüro", layout="centered")

# Modell laden
@st.cache_resource
def load_ki_model():
    return load_model("keras_Model.h5", compile=False)

model = load_ki_model()
class_names = open("labels.txt", "r").readlines()

np.set_printoptions(suppress=True)

FUNDE_DATEI = "funde.json"

# Datei erstellen falls nicht vorhanden
if not os.path.exists(FUNDE_DATEI):
    with open(FUNDE_DATEI, "w") as f:
        json.dump([], f)


def predict_image(image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score


def speichere_fund(tierart, confidence):
    with open(FUNDE_DATEI, "r") as f:
        daten = json.load(f)

    daten.append({
        "tierart": tierart,
        "confidence": round(confidence * 100, 2),
        "datum": datetime.now().strftime("%d.%m.%Y %H:%M")
    })

    with open(FUNDE_DATEI, "w") as f:
        json.dump(daten, f, indent=4)


def lade_funde():
    with open(FUNDE_DATEI, "r") as f:
        return json.load(f)


# ---------------- UI ----------------

st.title("🐾 KI Fundbüro")
st.write("Lade ein Bild hoch – die KI erkennt, ob es eine Katze oder ein Hund ist.")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("🔎 Analysieren"):
        class_name, confidence = predict_image(image)

        st.success(f"Erkannt: **{class_name}**")
        st.write(f"Sicherheit: {round(confidence*100,2)} %")

        speichere_fund(class_name, confidence)


st.divider()

st.subheader("🔍 Ich vermisse...")

gesucht = st.radio("Was vermisst du?", ["Katze", "Hund"])

if st.button("Neueste Funde anzeigen"):
    funde = lade_funde()

    passende_funde = [f for f in funde if gesucht.lower() in f["tierart"].lower()]

    if passende_funde:
        for fund in reversed(passende_funde[-5:]):
            st.write(f"🐾 {fund['tierart']} | {fund['confidence']} % | {fund['datum']}")
    else:
        st.warning("Keine passenden Funde gefunden.")
