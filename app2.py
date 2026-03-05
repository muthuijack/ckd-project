import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import hashlib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import joblib
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI CKD Monitoring System",
    layout="wide",
    page_icon="🏥"
)


# ==========================================
# MYSQL DATABASE CONNECTION
# ==========================================

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="68466450@machariamuthui",
    database="ckd_system"
)

cursor = conn.cursor()


# ==========================================
# CREATE TABLES
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE,
    password VARCHAR(255),
    role VARCHAR(20),
    phone VARCHAR(20)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100),
    model_used VARCHAR(100),
    probability FLOAT,
    prediction INT,
    visit_number INT,
    next_visit DATE,
    created_at DATETIME
)
""")

conn.commit()


# ==========================================
# LANGUAGE SYSTEM
# ==========================================

LANGUAGES = {

    "English": {

        "about": "Chronic Kidney Disease (CKD) is a long-term condition where kidneys gradually lose their ability to filter waste from the blood.",

        "low": "Low Risk - Maintain healthy lifestyle.",

        "moderate": "Moderate Risk - Regular monitoring required.",

        "high": "High Risk - Immediate medical attention advised.",

        "download": "Download Medical Report"
    },

    "French": {

        "about": "La maladie rénale chronique est une condition où les reins perdent progressivement leur fonction.",

        "low": "Risque faible - Maintenez un mode de vie sain.",

        "moderate": "Risque modéré - Surveillance médicale recommandée.",

        "high": "Risque élevé - Consultation médicale urgente.",

        "download": "Télécharger le rapport médical"
    }
}


# ==========================================
# SECURITY
# ==========================================

def hash_password(password):

    return hashlib.sha256(password.encode()).hexdigest()


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_models():

    rf = joblib.load("ckd_random_forest.pkl") if os.path.exists("ckd_random_forest.pkl") else None

    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None

    return rf, scaler


rf_model, scaler = load_models()


# ==========================================
# SEVERITY GRAPH
# ==========================================

def show_severity(prob):

    percent = prob * 100

    fig, ax = plt.subplots()

    ax.barh(["Risk"], [percent])

    ax.set_xlim(0, 100)

    ax.set_title("CKD Severity Scale (%)")

    st.pyplot(fig)

    st.metric("Risk Probability", f"{percent:.2f}%")


# ==========================================
# FOLLOW UP PREDICTION
# ==========================================

def calculate_followup(prob, username):

    cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE username=%s",
        (username,)
    )

    visit_count = cursor.fetchone()[0]

    percent = prob * 100

    if percent < 30:

        days = 180

    elif percent < 70:

        days = 30

    else:

        days = 7

    next_visit = datetime.now() + timedelta(days=days)

    return visit_count + 1, next_visit.date()


# ==========================================
# SMS SIMULATION
# ==========================================

def send_sms(username, next_visit):

    cursor.execute(
        "SELECT phone FROM users WHERE username=%s",
        (username,)
    )

    phone = cursor.fetchone()

    if phone and phone[0]:

        st.success(f"📩 SMS Reminder sent to {phone[0]} for {next_visit}")


# ==========================================
# PDF REPORT
# ==========================================

def generate_pdf(username, prob, next_visit):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI CKD Medical Report", styles["Title"]))

    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Patient: {username}", styles["Normal"]))

    elements.append(Paragraph(f"Risk Probability: {prob*100:.2f}%", styles["Normal"]))

    elements.append(Paragraph(f"Next Visit: {next_visit}", styles["Normal"]))

    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Generated by AI CKD Monitoring System", styles["Italic"]))

    doc.build(elements)

    buffer.seek(0)

    return buffer


# ==========================================
# SESSION
# ==========================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


# ==========================================
# LOGIN / REGISTER PAGE
# ==========================================

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

                cursor.execute(
                    "INSERT INTO users (username,password,role,phone) VALUES (%s,%s,%s,%s)",
                    (username, hash_password(password), role, phone)
                )

                conn.commit()

                st.success("Registered successfully")

            except:

                st.error("Username already exists")


    if option == "Login":

        if st.button("Login"):

            cursor.execute(
                "SELECT * FROM users WHERE username=%s AND password=%s",
                (username, hash_password(password))
            )

            user = cursor.fetchone()

            if user:

                st.session_state.logged_in = True

                st.session_state.username = user[1]

                st.session_state.role = user[3]

                st.rerun()

            else:

                st.error("Invalid credentials")


# ==========================================
# PATIENT PAGE
# ==========================================

def patient_page():

    st.sidebar.title("Patient Panel")

    language = st.sidebar.selectbox("🌍 Select Language", list(LANGUAGES.keys()))

    text = LANGUAGES[language]


    st.title("🧪 CKD Risk Prediction")


    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("🩺 About CKD")

        st.write(text["about"])


    with col2:

        st.subheader("⚠ Common Causes")

        st.markdown("""

        - Diabetes  

        - High Blood Pressure  

        - Genetic disorders  

        - Kidney infections  

        """)


    with col3:

        st.subheader("🔍 Common Symptoms")

        st.markdown("""

        - Swelling in legs  

        - Fatigue  

        - Urination changes  

        - Nausea  

        """)


    st.markdown("---")

    st.subheader("Enter Medical Information")


    age = st.number_input("Age", 1, 120, 45)

    bp = st.number_input("Blood Pressure", 50, 200, 80)

    bgr = st.number_input("Blood Glucose", 50, 500, 120)

    bu = st.number_input("Blood Urea", 1, 400, 40)

    sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.2)

    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.5)


    if st.button("Predict"):

        df = pd.DataFrame([{

            "age": age,

            "bp": bp,

            "bgr": bgr,

            "bu": bu,

            "sc": sc,

            "hemo": hemo

        }])


        features = scaler.feature_names_in_

        aligned = pd.DataFrame(0, index=[0], columns=features)


        for col in df.columns:

            if col in features:

                aligned[col] = df[col]


        scaled = scaler.transform(aligned)

        prob = float(rf_model.predict_proba(scaled)[0][1])

        prob = max(0, min(1, prob))


        st.subheader("📊 Risk Analysis")

        show_severity(prob)


        visit_number, next_visit = calculate_followup(prob, st.session_state.username)


        cursor.execute("""

        INSERT INTO predictions

        (username,model_used,probability,prediction,visit_number,next_visit,created_at)

        VALUES (%s,%s,%s,%s,%s,%s,%s)

        """,

        (

            st.session_state.username,

            "Random Forest",

            prob,

            1 if prob > 0.5 else 0,

            visit_number,

            next_visit,

            datetime.now()

        ))

        conn.commit()


        send_sms(st.session_state.username, next_visit)


        if prob < 0.3:

            st.success(text["low"])

        elif prob < 0.7:

            st.warning(text["moderate"])

        else:

            st.error(text["high"])


        pdf = generate_pdf(st.session_state.username, prob, next_visit)


        st.download_button(

            label=text["download"],

            data=pdf,

            file_name="ckd_report.pdf",

            mime="application/pdf"

        )


# ==========================================
# DOCTOR DASHBOARD
# ==========================================

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


# ==========================================
# ROUTER
# ==========================================

if not st.session_state.logged_in:

    login_page()

else:

    if st.session_state.role == "patient":

        patient_page()

    else:

        doctor_page()

