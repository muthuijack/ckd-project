import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="AI CKD Monitoring System",
                   layout="wide",
                   page_icon="🏥")

# =============================
# DATABASE
# =============================
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

# =============================
# PASSWORD
# =============================
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

# =============================
# LOAD MODELS
# =============================
def load_models():
    dnn, rf, scaler = None, None, None

    if os.path.exists("ckd_dnn_model.keras"):
        dnn = load_model("ckd_dnn_model.keras")

    if os.path.exists("ckd_random_forest.pkl"):
        rf = joblib.load("ckd_random_forest.pkl")

    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")

    return dnn, rf, scaler

dnn_model, rf_model, scaler = load_models()

# =============================
# FOLLOW-UP LOGIC
# =============================
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

    # Disease progression check
    previous = cursor.execute("""
        SELECT probability FROM predictions
        WHERE username=?
        ORDER BY created_at DESC
        LIMIT 1
    """, (username,)).fetchone()

    if previous:
        previous_prob = previous[0]
        if prob - previous_prob > 0.15:
            days = 7  # escalate if worsening rapidly

    next_visit = datetime.now() + timedelta(days=days)

    return visit_count + 1, next_visit.date()

# =============================
# SMS REMINDER (SIMULATION)
# =============================
def send_sms_reminder(username, next_visit):

    phone = cursor.execute(
        "SELECT phone FROM users WHERE username=?",
        (username,)
    ).fetchone()

    if phone and phone[0]:
        st.success(f"📩 SMS Reminder sent to {phone[0]} for visit on {next_visit}")
    else:
        st.warning("No phone number registered. Cannot send SMS.")

# =============================
# SEVERITY GRAPH
# =============================
def show_severity(prob):

    percent = prob * 100
    fig, ax = plt.subplots()

    ax.axvspan(0, 30)
    ax.axvspan(30, 70)
    ax.axvspan(70, 100)

    ax.axvline(percent)

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Risk (%)")
    ax.set_title("CKD Severity Scale")

    st.pyplot(fig)
    st.metric("Risk Level", f"{percent:.2f}%")

# =============================
# SESSION
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =============================
# LOGIN / REGISTER
# =============================
def login_page():

    st.title("🏥 AI CKD Monitoring System")

    option = st.radio("Select", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        role = st.selectbox("Role", ["patient", "doctor"])
        phone = st.text_input("Phone Number (for SMS reminders)")

        if st.button("Register"):
            try:
                cursor.execute("""
                INSERT INTO users (username,password,role,phone)
                VALUES (?,?,?,?)
                """, (username, hash_password(password), role, phone))
                conn.commit()
                st.success("Registered Successfully")
            except:
                st.error("Username already exists")

    if option == "Login":
        if st.button("Login"):
            user = cursor.execute("""
            SELECT * FROM users
            WHERE username=? AND password=?
            """, (username, hash_password(password))).fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.session_state.role = user[3]
                st.rerun()
            else:
                st.error("Invalid credentials")

# =============================
# PATIENT PAGE
# =============================
def patient_page():

    st.sidebar.title("Patient Panel")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.header("CKD Risk Assessment")

    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    bgr = st.number_input("Blood Glucose", 50, 500, 120)
    bu = st.number_input("Blood Urea", 1, 400, 40)
    sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.2)
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.5)

    if st.button("Predict"):

        if scaler is None:
            st.error("Scaler missing.")
            return

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

        if dnn_model:
            prob = float(dnn_model.predict(scaled, verbose=0)[0][0])
        else:
            prob = float(rf_model.predict_proba(scaled)[0][1])

        prob = max(0, min(1, prob))
        pred = 1 if prob > 0.5 else 0

        show_severity(prob)

        visit_number, next_visit = calculate_followup(prob, st.session_state.username)

        cursor.execute("""
        INSERT INTO predictions
        (username,model_used,probability,prediction,
         visit_number,next_visit,created_at)
        VALUES (?,?,?,?,?,?,?)
        """, (st.session_state.username,
              "DNN",
              prob,
              pred,
              visit_number,
              str(next_visit),
              datetime.now().isoformat()))
        conn.commit()

        st.info(f"📅 Recommended Next Visit: {next_visit}")
        st.write(f"Visit Number: {visit_number}")

        send_sms_reminder(st.session_state.username, next_visit)

# =============================
# DOCTOR PAGE
# =============================
def doctor_page():

    st.sidebar.title("Doctor Dashboard")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    data = pd.read_sql("""
        SELECT username, probability,
               visit_number, next_visit, created_at
        FROM predictions
        ORDER BY created_at DESC
    """, conn)

    if data.empty:
        st.info("No records yet.")
        return

    st.dataframe(data, use_container_width=True)

    st.subheader("Risk Distribution")
    st.bar_chart(data["probability"])

    st.subheader("Disease Progression Trends")
    st.line_chart(data.sort_values("created_at")["probability"])

# =============================
# ROUTER
# =============================
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.role == "patient":
        patient_page()
    else:
        doctor_page()
