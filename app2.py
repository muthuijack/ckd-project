import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import joblib
from tensorflow.keras.models import load_model

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="CKD Prediction System", layout="centered")

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
        full_name TEXT,
        email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        model_used TEXT,
        probability REAL,
        prediction INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

create_tables()

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

# =====================================================
# OFFICIAL FEATURE LIST (SAFE)
# =====================================================
FEATURES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv",
    "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"
]

# =====================================================
# UI
# =====================================================
st.title("🩺 Chronic Kidney Disease Prediction System")
st.write("Predict CKD using **Deep Neural Network** or **Random Forest**")

model_choice = st.radio(
    "Select Prediction Model",
    ["Deep Neural Network (DNN)", "Random Forest"]
)

st.divider()

# =====================================================
# PATIENT INFO
# =====================================================
st.subheader("Patient Information")
full_name = st.text_input("Full Name")
email = st.text_input("Email")

st.subheader("Medical Test Results")

def user_input():
    data = {
        "age": st.number_input("Age", 1, 120, 45),
        "bp": st.number_input("Blood Pressure", 50, 200, 80),
        "sg": st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025]),
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

df = df.reindex(columns=FEATURES)
df_scaled = scaler.transform(df)

# =====================================================
# PDF REPORT
# =====================================================
def generate_pdf(name, email, model, prob, pred, data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Chronic Kidney Disease Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Patient:</b> {name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Email:</b> {email}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Model Used:</b> {model}", styles["Normal"]))
    elements.append(Paragraph(
        f"<b>Prediction:</b> {'CKD Detected' if pred else 'No CKD'}",
        styles["Normal"]
    ))
    elements.append(Paragraph(f"<b>Probability:</b> {prob:.2%}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Feature", "Value"]] + [[k, str(v)] for k, v in data.items()]
    elements.append(Table(table_data))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔍 Predict CKD"):

    if model_choice == "Deep Neural Network (DNN)":
        prob = float(dnn_model.predict(df_scaled)[0][0])
        pred = 1 if prob > 0.5 else 0
    else:
        prob = float(rf_model.predict_proba(df_scaled)[0][1])
        pred = int(rf_model.predict(df_scaled)[0])

    # Save
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO patients (full_name, email) VALUES (?, ?)",
        (full_name, email)
    )
    patient_id = cur.lastrowid

    cur.execute(
        """
        INSERT INTO predictions (patient_id, model_used, probability, prediction)
        VALUES (?, ?, ?, ?)
        """,
        (patient_id, model_choice, prob, pred)
    )

    conn.commit()
    conn.close()

    # Result
    if pred:
        st.error(f"⚠️ CKD Detected (Probability: {prob:.2%})")
    else:
        st.success(f"✅ No CKD Detected (Probability: {1-prob:.2%})")

    pdf = generate_pdf(
        full_name, email, model_choice, prob, pred, df.iloc[0].to_dict()
    )

    st.download_button(
        "📄 Download Medical Report (PDF)",
        data=pdf,
        file_name="ckd_report.pdf",
        mime="application/pdf"
    )
