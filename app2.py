import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="centered"
)

# =====================================================
# MULTI-LANGUAGE TEXT
# =====================================================
TEXT = {
    "English": {
        "title": "🩺 Chronic Kidney Disease Prediction System",
        "patient_info": "Patient Information",
        "medical": "Medical Test Results",
        "predict": "🔍 Predict CKD",
        "ckd": "⚠️ CKD Detected",
        "no_ckd": "✅ No CKD Detected",
        "download": "📄 Download Medical Report (PDF)",
        "report_title": "Chronic Kidney Disease Prediction Report",
        "model": "Model Used",
        "prediction": "Prediction",
        "probability": "Probability"
    },
    "Swahili": {
        "title": "🩺 Mfumo wa Utambuzi wa Ugonjwa wa Figo",
        "patient_info": "Taarifa za Mgonjwa",
        "medical": "Vipimo vya Maabara",
        "predict": "🔍 Tambua Ugonjwa",
        "ckd": "⚠️ Ugonjwa wa Figo Umegunduliwa",
        "no_ckd": "✅ Hakuna Ugonjwa wa Figo",
        "download": "📄 Pakua Ripoti ya PDF",
        "report_title": "Ripoti ya Utambuzi wa Ugonjwa wa Figo",
        "model": "Mfumo Uliotumika",
        "prediction": "Matokeo",
        "probability": "Uwezekano"
    },
    "French": {
        "title": "🩺 Système de Prédiction des Maladies Rénales",
        "patient_info": "Informations du Patient",
        "medical": "Résultats Médicaux",
        "predict": "🔍 Prédire la Maladie",
        "ckd": "⚠️ Maladie Rénale Détectée",
        "no_ckd": "✅ Aucune Maladie Rénale",
        "download": "📄 Télécharger le Rapport PDF",
        "report_title": "Rapport de Prédiction de Maladie Rénale",
        "model": "Modèle Utilisé",
        "prediction": "Résultat",
        "probability": "Probabilité"
    }
}

language = st.selectbox("🌍 Language / Lugha / Langue", list(TEXT.keys()))
T = TEXT[language]

# =====================================================
# LOAD MODEL & SCALER (SAFE)
# =====================================================
MODEL_PATH = "ckd_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model files missing. Ensure ckd_model.pkl & scaler.pkl are in the repo.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = list(scaler.feature_names_in_)

# =====================================================
# UI
# =====================================================
st.title(T["title"])

# =====================================================
# PATIENT INFO
# =====================================================
st.subheader(T["patient_info"])
name = st.text_input("Full Name")
email = st.text_input("Email")

st.subheader(T["medical"])

# =====================================================
# USER INPUTS
# =====================================================
def user_input():
    data = {
        "age": st.number_input("Age", 1, 120, 45),
        "bp": st.number_input("Blood Pressure", 50, 200, 80),
        "sg": st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025]),
        "al": st.selectbox("Albumin", [0,1,2,3,4,5]),
        "su": st.selectbox("Sugar", [0,1,2,3,4,5]),
        "rbc": st.selectbox("Red Blood Cells", ["normal","abnormal"]),
        "pc": st.selectbox("Pus Cell", ["normal","abnormal"]),
        "pcc": st.selectbox("Pus Cell Clumps", ["notpresent","present"]),
        "ba": st.selectbox("Bacteria", ["notpresent","present"]),
        "bgr": st.number_input("Blood Glucose Random", 50, 500, 120),
        "bu": st.number_input("Blood Urea", 1, 400, 40),
        "sc": st.number_input("Serum Creatinine", 0.1, 20.0, 1.2),
        "sod": st.number_input("Sodium", 100, 200, 135),
        "pot": st.number_input("Potassium", 2.0, 10.0, 4.5),
        "hemo": st.number_input("Hemoglobin", 3.0, 20.0, 13.5),
        "pcv": st.number_input("Packed Cell Volume", 10, 60, 40),
        "wc": st.number_input("White Blood Cell Count", 3000, 20000, 8000),
        "rc": st.number_input("Red Blood Cell Count", 2.0, 8.0, 4.5),
        "htn": st.selectbox("Hypertension", ["no","yes"]),
        "dm": st.selectbox("Diabetes Mellitus", ["no","yes"]),
        "cad": st.selectbox("Coronary Artery Disease", ["no","yes"]),
        "appet": st.selectbox("Appetite", ["good","poor"]),
        "pe": st.selectbox("Pedal Edema", ["no","yes"]),
        "ane": st.selectbox("Anemia", ["no","yes"])
    }
    return pd.DataFrame([data])

df = user_input()

# =====================================================
# ENCODING
# =====================================================
binary_map = {
    "normal": 0, "abnormal": 1,
    "no": 0, "yes": 1,
    "notpresent": 0, "present": 1,
    "good": 0, "poor": 1
}

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].map(binary_map)

df = df[FEATURES]
df_scaled = scaler.transform(df)

# =====================================================
# PDF REPORT
# =====================================================
def generate_pdf(name, email, prob, pred):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{T['report_title']}</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"{T['model']}: ML Model", styles["Normal"]))
    elements.append(Paragraph(f"{T['prediction']}: {'CKD' if pred else 'No CKD'}", styles["Normal"]))
    elements.append(Paragraph(f"{T['probability']}: {prob:.2%}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Feature", "Value"]] + [[k, str(v)] for k, v in df.iloc[0].to_dict().items()]
    elements.append(Table(table_data))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# PREDICTION
# =====================================================
if st.button(T["predict"]):
    prob = float(model.predict_proba(df_scaled)[0][1])
    pred = 1 if prob > 0.5 else 0

    pdf = generate_pdf(name, email, prob, pred)

    if pred:
        st.error(f"{T['ckd']} ({prob:.2%})")
    else:
        st.success(f"{T['no_ckd']} ({1 - prob:.2%})")

    st.download_button(
        T["download"],
        data=pdf,
        file_name="ckd_report.pdf",
        mime="application/pdf"
    )
