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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CKD Hospital System",
    layout="wide",
    page_icon="🏥"
)

# =====================================================
# CUSTOM THEME STYLE
# =====================================================
st.markdown("""
<style>
.main {background-color: #f5f7fa;}
.sidebar .sidebar-content {background-color: #0e1117;}
h1, h2, h3 {color: #0e1117;}
.stButton>button {
    background-color: #0066cc;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #004999;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE
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
# PASSWORD HASH
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================================
# SESSION
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

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
# DANGER METER FUNCTION
# =====================================================
def show_danger_meter(prob):

    percent = prob * 100

    if percent < 30:
        color = "🟢 Low Risk"
    elif percent < 70:
        color = "🟡 Moderate Risk"
    else:
        color = "🔴 High Risk"

    st.subheader("Risk Level")
    st.progress(percent / 100)
    st.metric("CKD Probability", f"{percent:.2f}%")
    st.markdown(f"### {color}")

# =====================================================
# PDF
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
    elements.append(Paragraph(
        f"Result: {'CKD Detected' if prediction==1 else 'No CKD'}",
        styles["Normal"]
    ))
    elements.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# LOGIN PAGE
# =====================================================
def login_page():
    st.title("🏥 CKD Hospital Login System")

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

# =====================================================
# PATIENT PAGE
# =====================================================
def patient_page():

    st.sidebar.title("🧑 Patient Panel")
    st.sidebar.write(f"Welcome {st.session_state.username}")

    model_choice = st.sidebar.radio(
        "Model Selection",
        ["Deep Neural Network (DNN)", "Random Forest"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🩺 CKD Prediction Form")

    df = pd.DataFrame([{
        "age": st.number_input("Age", 1, 120, 45),
        "bp": st.number_input("Blood Pressure", 50, 200, 80),
        "bgr": st.number_input("Blood Glucose", 50, 500, 120),
        "bu": st.number_input("Blood Urea", 1, 400, 40),
        "sc": st.number_input("Serum Creatinine", 0.1, 20.0, 1.2),
        "hemo": st.number_input("Hemoglobin", 3.0, 20.0, 13.5)
    }])

    df_aligned = pd.DataFrame(0, index=[0], columns=FEATURES)
    for col in df.columns:
        if col in FEATURES:
            df_aligned[col] = df[col]

    df_scaled = scaler.transform(df_aligned)

    if st.button("Predict"):

        if model_choice == "Deep Neural Network (DNN)":
            prob = float(dnn_model.predict(df_scaled, verbose=0)[0][0])
            pred = 1 if prob > 0.5 else 0
        else:
            prob = float(rf_model.predict_proba(df_scaled)[0][1])
            pred = int(rf_model.predict(df_scaled)[0])

        show_danger_meter(prob)

        conn.execute("""
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
            "Download Medical Report",
            pdf,
            file_name="CKD_Report.pdf",
            mime="application/pdf"
        )

# =====================================================
# DOCTOR PAGE
# =====================================================
def doctor_page():

    st.sidebar.title("👩‍⚕️ Doctor Dashboard")
    st.sidebar.write(f"Dr. {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("📊 Patient Prediction Records")

    data = pd.read_sql("""
        SELECT username, model_used, probability, prediction, created_at
        FROM predictions
        ORDER BY created_at DESC
    """, conn)

    if data.empty:
        st.info("No patient records yet.")
    else:
        st.dataframe(data, use_container_width=True)

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
