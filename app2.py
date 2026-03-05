import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import joblib
import io

# PDF Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# =====================================================
# PAGE CONFIG & THEME
# =====================================================
st.set_page_config(
    page_title="NephroAI | Advanced CKD Portal",
    layout="wide",
    page_icon="🏥"
)

# Custom CSS for a Professional Medical Look
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #002b5c; }
    [data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE SYSTEM
# =====================================================
DB_FILE = "ckd_hospital.db"

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE, password TEXT, role TEXT, phone TEXT, created_at TEXT
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT, model_used TEXT, probability REAL, prediction INTEGER,
        visit_number INTEGER, next_visit TEXT, created_at TEXT,
        age INTEGER, bp INTEGER, bgr INTEGER, bu INTEGER, sc REAL, hemo REAL
    )""")
    conn.commit()
    return conn, cursor

conn, cursor = init_db()

# =====================================================
# UTILITIES & ML LOADING
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@st.cache_resource
def load_assets():
    model = joblib.load("ckd_random_forest.pkl") if os.path.exists("ckd_random_forest.pkl") else None
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return model, scaler

rf_model, scaler = load_assets()

# =====================================================
# PDF REPORT GENERATOR
# =====================================================
def generate_enhanced_pdf(username, prob, next_visit, metrics):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("OFFICIAL MEDICAL REPORT: CKD ANALYSIS", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Patient Name: {username}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Data Table
    data = [["Metric", "Value"],
            ["Age", metrics['age']],
            ["Blood Pressure", metrics['bp']],
            ["Serum Creatinine", metrics['sc']],
            ["Hemoglobin", metrics['hemo']],
            ["Risk Probability", f"{prob*100:.2f}%"]]
    
    t = Table(data, colWidths=[2*inch, 2*inch])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.dodgerblue),
                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                           ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Recommended Follow-up: {next_visit}", styles["Heading3"]))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# LOGIN / REGISTER UI
# =====================================================
def auth_page():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=150)
        st.title("NephroAI Portal")
        st.write("Advanced Kidney Health Monitoring using Machine Learning.")
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            u = st.text_input("Username", key="l_u")
            p = st.text_input("Password", type="password", key="l_p")
            if st.button("Sign In"):
                user = cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_password(p))).fetchone()
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username, st.session_state.role = user[1], user[3]
                    st.rerun()
                else: st.error("Invalid Credentials")
        
        with tab2:
            new_u = st.text_input("New Username")
            new_p = st.text_input("New Password", type="password")
            role = st.selectbox("I am a...", ["patient", "doctor"])
            phone = st.text_input("Mobile Number")
            if st.button("Create Account"):
                try:
                    cursor.execute("INSERT INTO users(username,password,role,phone,created_at) VALUES(?,?,?,?,?)",
                                 (new_u, hash_password(new_p), role, phone, str(datetime.now())))
                    conn.commit()
                    st.success("Account Created! Please Login.")
                except: st.error("User already exists.")

# =====================================================
# PATIENT INTERFACE
# =====================================================
def patient_dashboard():
    st.sidebar.header(f"Welcome, {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # Education Header
    

    st.title("🔍 Kidney Risk Assessment")
    
    with st.expander("Why are these values important?"):
        st.write("Serum Creatinine and Hemoglobin are critical indicators of how well your kidneys filter waste.")

    # Input Section
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 110, 45)
        bp = st.number_input("Blood Pressure (systolic)", 50, 200, 120)
        bgr = st.number_input("Blood Glucose (mg/dl)", 50, 500, 110)
    with col2:
        bu = st.number_input("Blood Urea", 1, 400, 30)
        sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.1)
        hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, 14.0)

    if st.button("Run AI Analysis"):
        if rf_model and scaler:
            # 1. Feature Alignment (Crucial Step)
            features = scaler.feature_names_in_
            input_df = pd.DataFrame(0, index=[0], columns=features)
            
            # Map UI inputs to dataframe
            input_map = {"age":age, "bp":bp, "bgr":bgr, "bu":bu, "sc":sc, "hemo":hemo}
            for k, v in input_map.items():
                if k in features: input_df[k] = v
            
            # 2. Prediction
            scaled_data = scaler.transform(input_df)
            prob = rf_model.predict_proba(scaled_data)[0][1]
            
            # 3. Logic for Follow-up
            days = 180 if prob < 0.3 else (30 if prob < 0.7 else 7)
            next_v = (datetime.now() + timedelta(days=days)).date()
            
            # 4. Visuals
            st.markdown("---")
            c_res1, c_res2 = st.columns([1, 2])
            with c_res1:
                st.metric("Risk Score", f"{prob*100:.1f}%")
                if prob > 0.7: st.error("High Risk Detected")
                elif prob > 0.3: st.warning("Moderate Risk Detected")
                else: st.success("Low Risk Detected")
            
            with c_res2:
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.barh(["Risk Level"], [prob], color='red' if prob > 0.5 else 'green')
                ax.set_xlim(0, 1)
                st.pyplot(fig)

            # 5. Save and PDF
            cursor.execute("""INSERT INTO predictions 
                (username, model_used, probability, prediction, visit_number, next_visit, created_at, age, bp, bgr, bu, sc, hemo)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", 
                (st.session_state.username, "Random Forest", prob, 1 if prob > 0.5 else 0, 1, str(next_v), str(datetime.now()), age, bp, bgr, bu, sc, hemo))
            conn.commit()
            
            pdf = generate_enhanced_pdf(st.session_state.username, prob, next_v, input_map)
            st.download_button("📩 Download Medical Report", data=pdf, file_name="CKD_Analysis.pdf")
        else:
            st.error("Model files missing. Please contact administrator.")

# =====================================================
# DOCTOR INTERFACE
# =====================================================
def doctor_dashboard():
    st.title("👨‍⚕️ Clinical Supervisor Dashboard")
    
    data = pd.read_sql("SELECT * FROM predictions", conn)
    
    if not data.empty:
        # High Level Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Patients Screened", len(data['username'].unique()))
        m2.metric("High Risk Alerts", len(data[data['probability'] > 0.7]))
        m3.metric("Avg Risk Score", f"{data['probability'].mean()*100:.1f}%")
        
        st.subheader("Patient Records")
        st.dataframe(data.sort_values(by="created_at", ascending=False), use_container_width=True)
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.subheader("Risk Distribution")
            st.bar_chart(data['probability'])
        with col_plot2:
            st.subheader("Age vs Risk Factor")
            st.scatter_chart(data, x='age', y='probability', color='prediction')
    else:
        st.info("No clinical data recorded yet.")

# =====================================================
# MAIN ROUTING
# =====================================================
if __name__ == "__main__":
    if not st.session_state.get("logged_in"):
        auth_page()
    else:
        if st.session_state.role == "patient":
            patient_dashboard()
        else:
            doctor_dashboard()

