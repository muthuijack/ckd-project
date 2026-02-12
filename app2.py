import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide"
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

language = st.sidebar.selectbox("🌍 Language / Lugha / Langue", list(TEXT.keys()))
T = TEXT[language]

# =====================================================
# DATABASE
# =====================================================
def get_connection():
    return sqlite3.connect("ckd.db", check_same_thread=False)

def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        language TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        model_used TEXT,
        probability REAL,
        prediction INTEGER,
        created_at TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    """)

    conn.commit()
    conn.close()

create_tables()

# =====================================================
# LOAD MODEL & SCALER (UNCHANGED)
# =====================================================
MODEL_PATH = "ckd_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = list(scaler.feature_names_in_)

# =====================================================
# UI
# =====================================================
st.title(T["title"])

model_choice = st.radio(
    "Select Prediction Model",
    ["Random Forest"]  # keeping your working setup
)

# =====================================================
# PATIENT INFO
# =====================================================
st.subheader(T["patient_info"])
name = st.text_input("Full Name")
email = st.text_input("Email")

st.subheader(T["medical"])

# =====================================================
# USER INPUT
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
def generate_pdf(prob, pred):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{T['report_title']}</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"{T['model']}: Random Forest", styles["Normal"]))
    elements.append(Paragraph(f"{T['prediction']}: {'CKD' if pred else 'No CKD'}", styles["Normal"]))
    elements.append(Paragraph(f"{T['probability']}: {prob:.2%}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Feature", "Value"]] + [[k, str(v)] for k, v in df.iloc[0].to_dict().items()]
    elements.append(Table(table_data))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# PREDICTION + SAVE
# =====================================================
if st.button(T["predict"]):
    prob = float(model.predict_proba(df_scaled)[0][1])
    pred = 1 if prob > 0.5 else 0

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO patients (name, email, language, created_at) VALUES (?, ?, ?, ?)",
        (name, email, language, datetime.now().isoformat())
    )
    pid = cur.lastrowid

    cur.execute(
        """
        INSERT INTO predictions
        (patient_id, model_used, probability, prediction, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (pid, "Random Forest", prob, pred, datetime.now().isoformat())
    )

    conn.commit()
    conn.close()

    pdf = generate_pdf(prob, pred)

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

# =====================================================
# ADMIN DASHBOARD
# =====================================================
st.divider()
st.header("📊 Admin Dashboard")

conn = get_connection()
data = pd.read_sql("""
    SELECT p.name, p.email, p.language,
           pr.model_used, pr.probability, pr.prediction, pr.created_at
    FROM predictions pr
    JOIN patients p ON p.patient_id = pr.patient_id
    ORDER BY pr.created_at DESC
""", conn)
conn.close()

if not data.empty:

    # FILTERS
    col1, col2 = st.columns(2)
    with col1:
        lang_filter = st.multiselect("Filter by Language", data["language"].unique())
    with col2:
        model_filter = st.multiselect("Filter by Model", data["model_used"].unique())

    if lang_filter:
        data = data[data["language"].isin(lang_filter)]
    if model_filter:
        data = data[data["model_used"].isin(model_filter)]

    st.dataframe(data, use_container_width=True)

    # EXPORT
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Export CSV", csv, "ckd_predictions.csv", "text/csv")

    # ANALYTICS
    st.subheader("📈 Analytics")

    colA, colB = st.columns(2)
    with colA:
        st.metric("Total Predictions", len(data))
        st.metric("CKD Detected", int(data["prediction"].sum()))

    with colB:
        st.metric("No CKD", int((data["prediction"] == 0).sum()))
        st.metric("Average Risk", f"{data['probability'].mean():.2%}")

    fig, ax = plt.subplots()
    data["prediction"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Prediction Distribution")
    ax.set_xticklabels(["No CKD", "CKD"], rotation=0)
    st.pyplot(fig)

else:
    st.info("No records yet.")
