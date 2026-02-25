import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import os
from datetime import datetime
from io import BytesIO
import joblib
from tensorflow.keras.models import load_model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CKD Hospital System",
    layout="wide",
    page_icon="🏥"
)

# =====================================================
# STYLING
# =====================================================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.stButton>button {
    background-color: #0066cc;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("ckd.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

cursor.execute("""
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
# PASSWORD HASH
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================================
# SAFE MODEL LOADING
# =====================================================
def load_models():

    dnn_model = None
    rf_model = None
    scaler = None

    if os.path.exists("ckd_dnn_model.keras"):
        dnn_model = load_model("ckd_dnn_model.keras")

    if os.path.exists("ckd_random_forest.pkl"):
        rf_model = joblib.load("ckd_random_forest.pkl")

    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")

    return dnn_model, rf_model, scaler

dnn_model, rf_model, scaler = load_models()

# =====================================================
# DANGER METER
# =====================================================
def show_danger(prob):

    percent = prob * 100

    if percent < 30:
        label = "🟢 LOW RISK"
    elif percent < 70:
        label = "🟡 MODERATE RISK"
    else:
        label = "🔴 HIGH RISK"

    st.progress(percent/100)
    st.metric("CKD Probability", f"{percent:.2f}%")
    st.markdown(f"### {label}")

# =====================================================
# PDF REPORT
# =====================================================
def generate_pdf(username, prob, pred):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("CKD Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Patient: {username}", styles["Normal"]))
    elements.append(Paragraph(f"Probability: {prob:.2%}", styles["Normal"]))
    elements.append(Paragraph(
        f"Result: {'CKD Detected' if pred==1 else 'No CKD'}",
        styles["Normal"]
    ))
    elements.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# SESSION
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =====================================================
# LOGIN PAGE
# =====================================================
def login_page():

    st.title("🏥 CKD Hospital Login")

    option = st.radio("Select Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        role = st.selectbox("Role", ["patient", "doctor"])
        if st.button("Register"):
            try:
                cursor.execute(
                    "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                    (username, hash_password(password), role)
                )
                conn.commit()
                st.success("Registered successfully.")
            except:
                st.error("Username already exists.")

    if option == "Login":
        if st.button("Login"):
            user = cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, hash_password(password))
            ).fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.session_state.role = user[3]
                st.rerun()
            else:
                st.error("Invalid credentials.")

# =====================================================
# PATIENT PAGE
# =====================================================
def patient_page():

    st.sidebar.title("🧑 Patient Panel")
    model_choice = st.sidebar.radio(
        "Model Selection",
        ["DNN", "Random Forest"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🩺 CKD Prediction Form")

    # FULL FEATURE INPUT
    df = pd.DataFrame([{
        "age": st.number_input("Age", 1, 120, 45),
        "bp": st.number_input("Blood Pressure", 50, 200, 80),
        "bgr": st.number_input("Blood Glucose", 50, 500, 120),
        "bu": st.number_input("Blood Urea", 1, 400, 40),
        "sc": st.number_input("Serum Creatinine", 0.1, 20.0, 1.2),
        "hemo": st.number_input("Hemoglobin", 3.0, 20.0, 13.5)
    }])

    if scaler is None:
        st.error("Scaler file missing.")
        return

    FEATURES = scaler.feature_names_in_
    df_aligned = pd.DataFrame(0, index=[0], columns=FEATURES)

    for col in df.columns:
        if col in FEATURES:
            df_aligned[col] = df[col]

    df_scaled = scaler.transform(df_aligned)

    if st.button("Predict"):

        if model_choice == "DNN" and dnn_model is not None:
            prob = float(dnn_model.predict(df_scaled, verbose=0)[0][0])
        elif model_choice == "Random Forest" and rf_model is not None:
            prob = float(rf_model.predict_proba(df_scaled)[0][1])
        else:
            st.error("Selected model not available.")
            return

        prob = max(0.0, min(1.0, prob))
        pred = 1 if prob > 0.5 else 0

        show_danger(prob)

        cursor.execute("""
        INSERT INTO predictions
        (username, model_used, probability, prediction, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (
            st.session_state.username,
            model_choice,
            prob,
            pred,
            datetime.now().isoformat()
        ))

        conn.commit()

        pdf = generate_pdf(st.session_state.username, prob, pred)

        st.download_button(
            "Download Report",
            pdf,
            file_name="CKD_Report.pdf",
            mime="application/pdf"
        )

# =====================================================
# DOCTOR PAGE
# =====================================================
def doctor_page():

    st.sidebar.title("👩‍⚕️ Doctor Dashboard")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("📊 Patient Predictions")

    data = pd.read_sql("""
        SELECT username, model_used, probability, prediction, created_at
        FROM predictions
        ORDER BY created_at DESC
    """, conn)

    if data.empty:
        st.info("No records yet.")
    else:
        st.dataframe(data, use_container_width=True)
        st.subheader("CKD Probability Overview")
        st.bar_chart(data["probability"])

# =====================================================
# ROUTER
# =====================================================
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.role == "patient":
        patient_page()
    else:
        doctor_page()
