import streamlit as st
import pandas as pd
import joblib
import sqlite3
import hashlib
from datetime import datetime
from io import BytesIO
from tensorflow.keras.models import load_model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="CKD Prediction System", layout="wide")

# =========================================
# DATABASE
# =========================================
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

# =========================================
# PASSWORD HASH
# =========================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =========================================
# SESSION STATE
# =========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

# =========================================
# LOAD MODELS
# =========================================
@st.cache_resource
def load_models():
    dnn = load_model("ckd_dnn_model.keras")
    rf = joblib.load("ckd_random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    return dnn, rf, scaler

dnn_model, rf_model, scaler = load_models()
FEATURES = list(scaler.feature_names_in_)

# =========================================
# PDF GENERATOR
# =========================================
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

# =========================================
# LOGIN PAGE
# =========================================
def login_page():
    st.title("🔐 CKD System Login")

    menu = st.radio("Select Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if menu == "Register":
        role = st.selectbox("Register as", ["patient", "doctor"])
        if st.button("Register"):
            try:
                conn.execute(
                    "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                    (username, hash_password(password), role)
                )
                conn.commit()
                st.success("Registered successfully.")
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
                st.session_state.username = user[1]
                st.session_state.role = user[3]
                st.rerun()
            else:
                st.error("Invalid credentials.")

# =========================================
# PATIENT SIDEBAR + PAGE
# =========================================
def patient_page():

    st.sidebar.title("🧑 Patient Panel")
    st.sidebar.write(f"Welcome, {st.session_state.username}")
    model_choice = st.sidebar.radio("Select Model",
                                     ["Deep Neural Network (DNN)", "Random Forest"])

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🩺 CKD Prediction Form")

    df = pd.DataFrame([{
        "age": st.number_input("Age", 1, 120, 45),
        "bp": st.number_input("Blood Pressure", 50, 200, 80),
        "sg": st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025]),
        "al": st.selectbox("Albumin", [0,1,2,3,4,5]),
        "su": st.selectbox("Sugar", [0,1,2,3,4,5]),
        "bgr": st.number_input("Blood Glucose Random", 50, 500, 120),
        "bu": st.number_input("Blood Urea", 1, 400, 40),
        "sc": st.number_input("Serum Creatinine", 0.1, 20.0, 1.2),
        "hemo": st.number_input("Hemoglobin", 3.0, 20.0, 13.5)
    }])

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

        st.success(f"Prediction Probability: {prob:.2%}")

        conn.execute(
            "INSERT INTO predictions VALUES (NULL, ?, ?, ?, ?, ?)",
            (st.session_state.username, model_choice, prob, pred,
             datetime.now().isoformat())
        )
        conn.commit()

        pdf = generate_pdf(st.session_state.username, prob, pred)

        st.download_button("Download Report",
                           pdf,
                           file_name="CKD_Report.pdf",
                           mime="application/pdf")

# =========================================
# DOCTOR SIDEBAR + DASHBOARD
# =========================================
def doctor_page():

    st.sidebar.title("👩‍⚕️ Doctor Dashboard")
    st.sidebar.write(f"Welcome, Dr. {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("📊 Patient Predictions Overview")

    data = pd.read_sql(
        "SELECT username, model_used, probability, prediction, created_at FROM predictions ORDER BY created_at DESC",
        conn
    )

    if data.empty:
        st.info("No patient records yet.")
    else:
        st.dataframe(data, use_container_width=True)

# =========================================
# ROUTER
# =========================================
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.role == "patient":
        patient_page()
    elif st.session_state.role == "doctor":
        doctor_page()
