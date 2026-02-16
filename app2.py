import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
from io import BytesIO
from tensorflow.keras.models import load_model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="CKD Prediction System", layout="wide")

# =====================================================
# SIDEBAR
# =====================================================
TEXT = {
    "English": {
        "title": "🩺 Chronic Kidney Disease Prediction System",
        "predict": "🔍 Predict CKD",
        "download": "📄 Download PDF Report"
    },
    "Swahili": {
        "title": "🩺 Mfumo wa Ugunduzi wa Ugonjwa wa Figo",
        "predict": "🔍 Tambua Ugonjwa",
        "download": "📄 Pakua Ripoti ya PDF"
    },
    "French": {
        "title": "🩺 Système de Prédiction des Maladies Rénales",
        "predict": "🔍 Prédire la Maladie",
        "download": "📄 Télécharger le Rapport PDF"
    }
}

language = st.sidebar.selectbox("🌍 Language", list(TEXT.keys()), key="lang")
model_choice = st.sidebar.radio(
    "🧠 Select Model",
    ["Deep Neural Network (DNN)", "Random Forest"],
    key="model_choice"
)

T = TEXT[language]

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    dnn = load_model("ckd_dnn_model.keras")
    rf = joblib.load("ckd_random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    return dnn, rf, scaler

dnn_model, rf_model, scaler = load_models()
FEATURES = list(scaler.feature_names_in_)

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("ckd.db", check_same_thread=False)
cursor = conn.cursor()

# Ensure schema is correct (drop and recreate for consistency)
cursor.execute("DROP TABLE IF EXISTS predictions")
cursor.execute("""
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    model_used TEXT,
    probability REAL,
    prediction INTEGER,
    created_at TEXT
)
""")
conn.commit()

# =====================================================
# MAIN UI
# =====================================================
st.title(T["title"])

st.subheader("Patient Information")
name = st.text_input("Full Name", key="name")
email = st.text_input("Email", key="email")

st.subheader("Medical Test Results")

# =====================================================
# INPUT FORM
# =====================================================
def user_input():
    return pd.DataFrame([{
        "age": st.number_input("Age", 1, 120, 45, key="age"),
        "bp": st.number_input("Blood Pressure", 50, 200, 80, key="bp"),
        "sg": st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025], key="sg"),
        "al": st.selectbox("Albumin", [0,1,2,3,4,5], key="al"),
        "su": st.selectbox("Sugar", [0,1,2,3,4,5], key="su"),
        "rbc": st.selectbox("Red Blood Cells", ["normal","abnormal"], key="rbc"),
        "pc": st.selectbox("Pus Cell", ["normal","abnormal"], key="pc"),
        "pcc": st.selectbox("Pus Cell Clumps", ["notpresent","present"], key="pcc"),
        "ba": st.selectbox("Bacteria", ["notpresent","present"], key="ba"),
        "bgr": st.number_input("Blood Glucose Random", 50, 500, 120, key="bgr"),
        "bu": st.number_input("Blood Urea", 1, 400, 40, key="bu"),
        "sc": st.number_input("Serum Creatinine", 0.1, 20.0, 1.2, key="sc"),
        "sod": st.number_input("Sodium", 100, 200, 135, key="sod"),
        "pot": st.number_input("Potassium", 2.0, 10.0, 4.5, key="pot"),
        "hemo": st.number_input("Hemoglobin", 3.0, 20.0, 13.5, key="hemo"),
        "pcv": st.number_input("Packed Cell Volume", 10, 60, 40, key="pcv"),
        "wc": st.number_input("White Blood Cell Count", 3000, 20000, 8000, key="wc"),
        "rc": st.number_input("Red Blood Cell Count", 2.0, 8.0, 4.5, key="rc"),
        "htn": st.selectbox("Hypertension", ["no","yes"], key="htn"),
        "dm": st.selectbox("Diabetes Mellitus", ["no","yes"], key="dm"),
        "cad": st.selectbox("Coronary Artery Disease", ["no","yes"], key="cad"),
        "appet": st.selectbox("Appetite", ["good","poor"], key="appet"),
        "pe": st.selectbox("Pedal Edema", ["no","yes"], key="pe"),
        "ane": st.selectbox("Anemia", ["no","yes"], key="ane")
    }])

df = user_input()

# =====================================================
# SAFE ENCODING
# =====================================================
binary_map = {
    "normal": 0, "abnormal": 1,
    "no": 0, "yes": 1,
    "notpresent": 0, "present": 1,
    "good": 0, "poor": 1
}

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.lower().map(binary_map)

df = df.fillna(0)

# =====================================================
# FEATURE ALIGNMENT
# =====================================================
df_aligned = pd.DataFrame(columns=FEATURES)
for col in FEATURES:
    df_aligned[col] = df[col] if col in df.columns else 0
df = df_aligned.apply(pd.to_numeric)

if df.isnull().values.any():
    st.error("Invalid input detected.")
    st.stop()

df_scaled = scaler.transform(df)

if not np.isfinite(df_scaled).all():
    st.error("Scaling produced invalid values.")
    st.stop()

# =====================================================
# RISK DISPLAY
# =====================================================
def display_risk(prob):
    if prob < 0.25:
        color, label = "#2ecc71", "LOW RISK"
    elif prob < 0.5:
        color, label = "#f1c40f", "MODERATE RISK"
    elif prob < 0.75:
        color, label = "#e67e22", "HIGH RISK"
    else:
        color, label = "#e74c3c", "CRITICAL RISK"

    st.markdown(f"""
        <div style="
            padding:20px;
            border-radius:12px;
            background-color:{color};
            text-align:center;
            font-size:22px;
            font-weight:bold;
            color:white;">
            {label} — {prob:.2%}
        </div>
    """, unsafe_allow_html=True)

# =====================================================
# PDF REPORT
# =====================================================
def generate_pdf(name, probability, prediction):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("CKD Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Patient Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Probability: {probability:.2%}", styles["Normal"]))
    elements.append(Paragraph(f"Prediction: {'CKD Detected' if prediction == 1 else 'No CKD'}", styles["Normal"]))
    elements.append(Paragraph(f"Generated: {datetime.now()}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# PREDICTION
# =====================================================
if st.button(T["predict"], key="predict_btn"):
    try:
        if model_choice == "Deep Neural Network (DNN)":
            prob = float(dnn_model.predict(df_scaled, verbose=0)[0][0])
            pred = 1 if prob > 0.5 else 0
        else:
            prob = float(rf_model.predict_proba(df_scaled)[0][1])
            pred = int(rf_model.predict(df_scaled)[0])

        if not np.isfinite(prob):
            st.error("Model returned invalid probability.")
            st.stop()

        display_risk(prob)

        cursor.execute(""
            INSERT INTO


