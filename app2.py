import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import joblib
import io

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI CKD Monitoring System",
    layout="wide",
    page_icon="🏥"
)

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
    role TEXT,
    phone TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    model_used TEXT,
    probability REAL,
    prediction INTEGER,
    visit_number INTEGER,
    next_visit TEXT,
    created_at TEXT
)
""")

conn.commit()

# =====================================================
# LANGUAGE SYSTEM
# =====================================================
LANGUAGES = {
    "English": {
        "about": "About Chronic Kidney Disease (CKD)",
        "desc": "CKD is a long-term condition where kidneys gradually lose function.",
        "low": "Low Risk - Maintain healthy lifestyle.",
        "moderate": "Moderate Risk - Regular monitoring required.",
        "high": "High Risk - Immediate medical attention advised.",
        "download": "Download Medical Report"
    },
    "French": {
        "about": "À propos de la Maladie Rénale Chronique",
        "desc": "La MRC est une condition où les reins perdent progressivement leur fonction.",
        "low": "Risque faible - Maintenez un mode de vie sain.",
        "moderate": "Risque modéré - Surveillance médicale recommandée.",
        "high": "Risque élevé - Consultation médicale urgente.",
        "download": "Télécharger le rapport médical"
    }
}

# =====================================================
# SECURITY
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================================
# LOAD MODELS (ONLY TABULAR)
# =====================================================
@st.cache_resource
def load_models():
    rf = joblib.load("ckd_random_forest.pkl") if os.path.exists("ckd_random_forest.pkl") else None
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return rf, scaler

rf_model, scaler = load_models()

# =====================================================
# SEVERITY GRAPH
# =====================================================
def show_severity(prob):
    percent = prob * 100
    fig, ax = plt.subplots()
    ax.barh(["Risk"], [percent])
    ax.set_xlim(0, 100)
    ax.set_title("CKD Severity Scale (%)")
    st.pyplot(fig)
    st.metric("Risk Probability", f"{percent:.2f}%")

# =====================================================
# FOLLOW-UP LOGIC
# =====================================================
def calculate_followup(prob, username):
    visit_count = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE username=?",
        (username,)
    ).fetchone()[0]

    percent = prob * 100

    if percent < 30:
        days = 180
    elif percent < 70:
        days = 30
    else:
        days = 7

    next_visit = datetime.now() + timedelta(days=days)

    return visit_count + 1, next_visit.date()

# =====================================================
# SMS SIMULATION
# =====================================================
def send_sms(username, next_visit):
    phone = cursor.execute(
        "SELECT phone FROM users WHERE username=?",
        (username,)
    ).fetchone()

    if phone and phone[0]:
        st.success(f"📩 SMS Reminder sent to {phone[0]} for {next_visit}")

# =====================================================
# PDF GENERATION
# =====================================================
def generate_pdf(username, prob, next_visit):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI CKD Medical Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Patient: {username}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Probability: {prob*100:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"Next Visit: {next_visit}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Generated by AI CKD Monitoring System", styles["Italic"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# SESSION
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =====================================================
# LOGIN / REGISTER
# =====================================================
def login_page():
    st.title("🏥 AI CKD Monitoring System")
    option = st.radio("Select Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        role = st.selectbox("Role", ["patient", "doctor"])
        phone = st.text_input("Phone")

        if st.button("Register"):
            try:
                cursor.execute("""
                INSERT INTO users (username,password,role,phone)
                VALUES (?,?,?,?)
                """, (username, hash_password(password), role, phone))
                conn.commit()
                st.success("Registered successfully")
            except:
                st.error("Username exists")

    if option == "Login":
        if st.button("Login"):
            user = cursor.execute("""
            SELECT * FROM users WHERE username=? AND password=?
            """, (username, hash_password(password))).fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.session_state.role = user[3]
                st.rerun()
            else:
                st.error("Invalid credentials")

# =====================================================
# PATIENT PAGE
# =====================================================
def patient_page():

    st.sidebar.title("Patient Panel")

    language = st.selectbox("Language", list(LANGUAGES.keys()))
    text = LANGUAGES[language]

    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    bgr = st.number_input("Blood Glucose", 50, 500, 120)
    bu = st.number_input("Blood Urea", 1, 400, 40)
    sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.2)
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.5)

    if st.button("Predict"):

        df = pd.DataFrame([{
            "age": age, "bp": bp, "bgr": bgr,
            "bu": bu, "sc": sc, "hemo": hemo
        }])

        features = scaler.feature_names_in_
        aligned = pd.DataFrame(0, index=[0], columns=features)

        for col in df.columns:
            if col in features:
                aligned[col] = df[col]

        scaled = scaler.transform(aligned)

        prob = float(rf_model.predict_proba(scaled)[0][1])
        prob = max(0, min(1, prob))

        show_severity(prob)

        visit_number, next_visit = calculate_followup(
            prob,
            st.session_state.username
        )

        cursor.execute("""
        INSERT INTO predictions
        (username,model_used,probability,prediction,
         visit_number,next_visit,created_at)
        VALUES (?,?,?,?,?,?,?)
        """, (
            st.session_state.username,
            "Random Forest",
            prob,
            1 if prob > 0.5 else 0,
            visit_number,
            str(next_visit),
            datetime.now().isoformat()
        ))
        conn.commit()

        send_sms(st.session_state.username, next_visit)

        st.subheader(text["about"])
        st.write(text["desc"])

        if prob < 0.3:
            st.success(text["low"])
        elif prob < 0.7:
            st.warning(text["moderate"])
        else:
            st.error(text["high"])

        pdf = generate_pdf(
            st.session_state.username,
            prob,
            next_visit
        )

        st.download_button(
            label=text["download"],
            data=pdf,
            file_name="ckd_report.pdf",
            mime="application/pdf"
        )

# =====================================================
# DOCTOR PAGE
# =====================================================
def doctor_page():
    st.title("Doctor Dashboard")

    data = pd.read_sql("SELECT * FROM predictions", conn)

    if data.empty:
        st.info("No records yet")
        return

    st.dataframe(data)

    st.subheader("Risk Distribution")
    st.bar_chart(data["probability"])

    st.subheader("Disease Progression")
    st.line_chart(data["probability"])

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
