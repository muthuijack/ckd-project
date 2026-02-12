import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="CKD Predictor", layout="centered")

MODEL_PATH = "ckd_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "age", "bp", "sg", "al", "su",
    "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc"
]

# -----------------------------
# LANGUAGE DICTIONARY
# -----------------------------
LANG = {
    "English": {
        "title": "Chronic Kidney Disease Predictor",
        "subtitle": "Enter patient details",
        "predict": "Predict",
        "result_ckd": "⚠️ CKD Detected",
        "result_no_ckd": "✅ No CKD Detected",
        "download": "Download PDF Report",
        "language": "Select Language",
        "report_title": "CKD Prediction Report",
        "date": "Date"
    },
    "Swahili": {
        "title": "Kikadirio cha Ugonjwa wa Figo",
        "subtitle": "Weka taarifa za mgonjwa",
        "predict": "Kadiria",
        "result_ckd": "⚠️ Ugonjwa wa figo umetambuliwa",
        "result_no_ckd": "✅ Hakuna ugonjwa wa figo",
        "download": "Pakua Ripoti ya PDF",
        "language": "Chagua Lugha",
        "report_title": "Ripoti ya Ugonjwa wa Figo",
        "date": "Tarehe"
    },
    "French": {
        "title": "Prédiction de Maladie Rénale",
        "subtitle": "Entrez les informations du patient",
        "predict": "Prédire",
        "result_ckd": "⚠️ Maladie rénale détectée",
        "result_no_ckd": "✅ Aucune maladie rénale détectée",
        "download": "Télécharger le rapport PDF",
        "language": "Choisir la langue",
        "report_title": "Rapport de Prédiction Rénale",
        "date": "Date"
    }
}

# -----------------------------
# LANGUAGE SELECTOR
# -----------------------------
language = st.sidebar.selectbox(
    "🌍 Language / Lugha / Langue",
    list(LANG.keys())
)

T = LANG[language]

# -----------------------------
# UI
# -----------------------------
st.title(T["title"])
st.subheader(T["subtitle"])

inputs = {}
for feature in FEATURES:
    inputs[feature] = st.number_input(
        feature.upper(),
        min_value=0.0,
        step=0.1
    )

# -----------------------------
# PREDICTION
# -----------------------------
if st.button(T["predict"]):
    df = pd.DataFrame([inputs])

    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]

    if prediction == 1:
        st.error(T["result_ckd"])
        result_text = T["result_ckd"]
    else:
        st.success(T["result_no_ckd"])
        result_text = T["result_no_ckd"]

    # -----------------------------
    # PDF GENERATION (UNICODE SAFE)
    # -----------------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Arial", "", "", uni=True)
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, T["report_title"], ln=True)
    pdf.cell(0, 10, f"{T['date']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)

    for k, v in inputs.items():
        pdf.cell(0, 8, f"{k.upper()}: {v}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, result_text, ln=True)

    pdf_path = "ckd_report.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            T["download"],
            f,
            file_name="CKD_Report.pdf",
            mime="application/pdf"
        )
