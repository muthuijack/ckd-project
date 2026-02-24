import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import hashlib
from datetime import datetime
from io import BytesIO
from tensorflow.keras.models import load_model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="CKD System", layout="wide")

# =====================================================
# DATABASE SETUP
# =====================================================
conn = sqlite3.connect("ckd.db", check_same_thread=False)

conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    model_used TEXT,
    probability REAL,
    prediction INTEGER,
    created_at TEXT
)
""")

conn.commit()

# =====================================================
# PASSWORD HASHING
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================================
# SESSION STATE
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

# =====================================================
# LOGIN / REGISTER PAGE
# =====================================================
def login_page():
    st.title("🔐 CKD System Login")

    menu = st.radio("Select Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if menu == "Register":
        role = st.selectbox("Register As", ["doctor", "patient"])

        if st.button("Register"):
            try:
                conn.execute(
                    "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                    (username, hash_password(password), role)
                )
                conn.commit()
                st.success("User registered successfully.")
            except:
                st.error("Username already exists.")

    if menu == "Login":
        if st.button("Login"):
            user = conn.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, hash_password(password))
            ).fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.role = user[3]
                st.session_state.username = user[1]
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid credentials.")

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
# PDF GENERATION
# =====================================================
def generate_pdf(username, probability, prediction):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("CKD Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Patient: {username}", styles["Normal"]))
    elements.append(Paragraph(f"Probability: {probability:.2%}", styles["Normal"]))
    elements.append(Paragraph(f"Result: {'CKD Detected' if prediction==1 else 'No CKD'}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# CKD PREDICTION PAGE
# =====================================================
def prediction_page():

    st.sidebar.write(f"Logged in as: {st.session_state.username} ({st.session_state.role})")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🩺 CKD Prediction")

    model_choice = st.sidebar.radio(
        "Select Model",
        ["Deep Neural Network (DNN)", "Random Forest"]
    )

    def user_input():
        return pd.DataFrame([{
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
        }])

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

    df = df.fillna(0)

    df_aligned = pd.DataFrame(columns=FEATURES)
    for col in FEATURES:
        df_aligned[col] = df[col] if col in df.columns else 0

    df_scaled = scaler.transform(df_aligned)

    if st.button("Predict"):
        if model_choice == "Deep Neural Network (DNN)":
            prob = float(dnn_model.predict(df_scaled, verbose=0)[0][0])
            pred = 1 if prob > 0.5 else 0
        else:
            prob = float(rf_model.predict_proba(df_scaled)[0][1])
            pred = int(rf_model.predict(df_scaled)[0])

        st.success(f"Probability: {prob:.2%}")

        conn.execute(
            "INSERT INTO predictions VALUES (NULL, ?, ?, ?, ?, ?)",
            (
                st.session_state.username,
                model_choice,
                prob,
                pred,
                datetime.now().isoformat()
            )
        )
        conn.commit()

        pdf = generate_pdf(st.session_state.username, prob, pred)

        st.download_button(
            "Download PDF Report",
            pdf,
            file_name="CKD_Report.pdf",
            mime="application/pdf"
        )

    # Doctor view all predictions
    if st.session_state.role == "doctor":
        st.subheader("📊 All Patient Predictions")
        data = pd.read_sql("SELECT * FROM predictions", conn)
        st.dataframe(data)

# =====================================================
# MAIN ROUTER
# =====================================================
if not st.session_state.logged_in:
    login_page()
else:
    prediction_page()
