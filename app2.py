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
# SIDEBAR (Language + Model)
# =====================================================
TEXT = {
    "English": {
        "title": "🩺 Chronic Kidney Disease Prediction System",
        "predict": "🔍 Predict CKD",
        "download": "📄 Download PDF Report"
    },
    "Swahili": {
        "title": "🩺 Mfumo wa Utambuzi wa Ugonjwa wa Figo",
        "predict": "🔍 Tambua Ugonjwa",
        "download": "📄 Pakua Ripoti ya PDF"
    },
    "French": {
        "title": "🩺 Système de Prédiction des Maladies Rénales",
        "predict": "🔍 Prédire la Maladie",
        "download": "📄 Télécharger le Rapport PDF"
    }
}

language = st.sidebar.selectbox("🌍 Language", list(TEXT.keys()), key="lang_select")
model_choice = st.sidebar.radio(
    "🧠 Select Model",
    ["Deep Neural Network (DNN)", "Random Forest"],
    key="model_select"
)

T = TEXT[language]

# =====================================================
# LOAD MODELS (ORIGINAL SAFE VERSION)
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
def get_connection():
    return sqlite3.connect("ckd.db", check_same_thread=False)

conn = get_connection()
conn.execute("""
CREATE TABLE IF NOT EXISTS predictions (
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
name = st.text_input("Full Name", key="patient_name")
email = st.text_input("Email", key="patient_email")

st.subheader("Medical Test Results")

# =====================================================
# USER INPUT WITH UNIQUE KEYS
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

df = df.reindex(columns=FEATURES)
df_scaled = scaler.transform(df)

# =====================================================
# RISK SCALE DISPLAY
# =====================================================
def display_risk(prob):
    if prob < 0.25:
        color, label = "green", "LOW RISK"
    elif prob < 0.5:
        color, label = "yellow", "MODERATE RISK"
    elif prob < 0.75:
        color, label = "orange", "HIGH RISK"
    else:
        color, label = "red", "CRITICAL RISK"

    st.markdown(f"""
    <div style="padding:15px;border-radius:10px;
                background-color:{color};
                text-align:center;
                font-size:22px;
                font-weight:bold;">
        {label} — {prob:.2%}
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================
if st.button(T["predict"], key="predict_button"):

    if model_choice == "Deep Neural Network (DNN)":
        prob = float(dnn_model.predict(df_scaled)[0][0])
        pred = 1 if prob > 0.5 else 0
    else:
        prob = float(rf_model.predict_proba(df_scaled)[0][1])
        pred = rf_model.predict(df_scaled)[0]

    display_risk(prob)

    conn.execute("""
        INSERT INTO predictions
        (name, email, model_used, probability, prediction, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, email, model_choice, prob, int(pred), datetime.now().isoformat()))
    conn.commit()

    st.success("Prediction saved to database.")
