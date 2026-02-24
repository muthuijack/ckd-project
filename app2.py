import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

st.set_page_config(page_title="CKD Prediction System", layout="wide")

# =====================================================
# LOAD MODEL (UNCHANGED STRUCTURE)
# =====================================================
model = pickle.load(open("ckd_model.pkl", "rb"))

# =====================================================
# DATABASE SETUP (AUTO SAFE)
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
# SIDEBAR STYLE
# =====================================================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.big-title {
    font-size:30px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# AUTH FUNCTIONS
# =====================================================
def login(username, password):
    user = cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    ).fetchone()
    return user

def register(username, password, role):
    try:
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, password, role)
        )
        conn.commit()
        return True
    except:
        return False

# =====================================================
# DANGER SCALE FUNCTION
# =====================================================
def danger_scale(prob):
    if prob < 0.33:
        color = "green"
        label = "LOW RISK"
    elif prob < 0.66:
        color = "orange"
        label = "MODERATE RISK"
    else:
        color = "red"
        label = "HIGH RISK"

    st.markdown(f"""
    <div style="padding:20px;border-radius:10px;background-color:{color};
                color:white;text-align:center;font-size:20px;">
        {label} <br> {prob*100:.2f}% probability
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PATIENT PAGE
# =====================================================
def patient_page():

    st.title("Patient Prediction Dashboard")

    age = st.number_input("Age", 1, 120)
    bp = st.number_input("Blood Pressure")
    sg = st.number_input("Specific Gravity")
    al = st.number_input("Albumin")
    su = st.number_input("Sugar")

    if st.button("Predict CKD"):

        input_data = np.array([[age, bp, sg, al, su]])

        try:
            prob = float(model.predict_proba(input_data)[0][1])
        except:
            prob = float(model.predict(input_data)[0])

        prob = max(0.0, min(1.0, prob))

        prediction = 1 if prob > 0.5 else 0

        danger_scale(prob)

        cursor.execute("""
        INSERT INTO predictions (username, model_used, probability, prediction, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (
            st.session_state.username,
            "CKD Model",
            prob,
            prediction,
            datetime.now().isoformat()
        ))
        conn.commit()

    st.subheader("Your Prediction History")

    data = pd.read_sql("""
        SELECT probability, prediction, created_at
        FROM predictions
        WHERE username=?
        ORDER BY created_at DESC
    """, conn, params=(st.session_state.username,))

    if not data.empty:
        st.dataframe(data)
    else:
        st.info("No predictions yet.")

# =====================================================
# DOCTOR PAGE
# =====================================================
def doctor_page():

    st.title("Doctor Dashboard")

    data = pd.read_sql("""
        SELECT username, model_used, probability, prediction, created_at
        FROM predictions
        ORDER BY created_at DESC
    """, conn)

    if data.empty:
        st.info("No patient records available.")
        return

    st.dataframe(data)

    st.subheader("CKD Risk Distribution")

    st.bar_chart(data["probability"])

# =====================================================
# MAIN APP FLOW
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("CKD Prediction Login")

    option = st.selectbox("Select Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        role = st.selectbox("Role", ["patient", "doctor"])
        if st.button("Register"):
            if register(username, password, role):
                st.success("Registered successfully!")
            else:
                st.error("Username already exists.")

    else:
        if st.button("Login"):
            user = login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.session_state.role = user[3]
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

else:

    st.sidebar.markdown(f"### Welcome {st.session_state.username}")
    st.sidebar.write("Role:", st.session_state.role)

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if st.session_state.role == "patient":
        patient_page()
    else:
        doctor_page()
