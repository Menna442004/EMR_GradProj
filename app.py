import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import io
import json
import soundfile as sf
import numpy as np
import whisper
import os
import torchvision.transforms as T
import torch
from PIL import Image
from datetime import datetime

from model import (
    predict, predict_topk,                          # NLP
    load_vision_engine, GradCAM,                    # Vision
    apply_clahe, encode_meta,                       # Vision helpers
    DEVICE, DISEASE_LABELS, OPTIMAL_THRESHOLDS,     # Vision constants
)

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Seha Track Pro", layout="wide", page_icon="🏥")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f8fafc; }
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
div.stButton > button {
    background: #2563eb; color: white;
    border-radius: 10px; border: none;
    padding: 0.5rem 1.5rem; font-weight: 500;
}
div.stButton > button:hover { background: #1d4ed8; }
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input { border-radius: 8px; }
.doctor-badge {
    background: #1e3a5f; color: #93c5fd;
    padding: 4px 12px; border-radius: 20px;
    font-size: 13px; display: inline-block; margin-bottom: 8px;
}
.section-card {
    background: white; border-radius: 14px;
    padding: 1.5rem; border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
/* X-ray specific additions */
.xray-positive {
    background: #fef2f2; border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 10px 14px;
    color: #991b1b; font-weight: 500; margin: 4px 0;
}
.xray-negative {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    border-radius: 8px; padding: 10px 14px;
    color: #166534; font-weight: 500; margin: 4px 0;
}
.xray-badge {
    display: inline-block; background: #eff6ff;
    border: 1px solid #93c5fd; color: #1d4ed8;
    border-radius: 16px; padding: 3px 12px;
    font-size: 12px; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

DB_PATH      = "patient_database.db"
WHISPER_CACHE = r"C:\Users\RofaR\OneDrive\Desktop\ManarGP\whisper_cache"
os.makedirs(WHISPER_CACHE, exist_ok=True)

# Val transform — must match notebook val_transform exactly
VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Database setup ──────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS doctors (
        doctor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name   TEXT,
        specialty   TEXT,
        created_at  TEXT
    );

    CREATE TABLE IF NOT EXISTS patients (
        patient_id      TEXT PRIMARY KEY,
        full_name       TEXT,
        date_of_birth   TEXT,
        age             INTEGER,
        gender          TEXT,
        blood_type      TEXT,
        phone           TEXT,
        email           TEXT,
        insurance_status TEXT,
        emergency_contact TEXT,
        allergies       TEXT,
        created_at      TEXT
    );

    CREATE TABLE IF NOT EXISTS visits (
        visit_id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id            TEXT REFERENCES patients(patient_id),
        doctor_id             INTEGER REFERENCES doctors(doctor_id),
        visit_datetime        TEXT,
        visit_type            TEXT,
        department            TEXT,
        chief_complaint       TEXT,
        asr_transcription     TEXT,
        symptom_top1          TEXT,
        symptom_top1_score    REAL,
        symptom_top2          TEXT,
        symptom_top2_score    REAL,
        symptom_top3          TEXT,
        symptom_top3_score    REAL,
        urgency               TEXT,
        severity_score        INTEGER,
        follow_up_needed      INTEGER,
        doctor_notes          TEXT,
        recommended_steps     TEXT,
        ground_truth_symptom  TEXT,
        audio_duration_sec    REAL,
        processed_at          TEXT
    );

    CREATE TABLE IF NOT EXISTS diagnoses (
        diagnosis_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id      INTEGER REFERENCES visits(visit_id),
        rank          INTEGER,
        disease       TEXT,
        likelihood    TEXT,
        reasoning     TEXT
    );

    CREATE TABLE IF NOT EXISTS xray_scans (
        scan_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      TEXT,
        patient_name    TEXT,
        doctor_id       INTEGER,
        scan_datetime   TEXT,
        patient_age     INTEGER,
        patient_gender  TEXT,
        view_position   TEXT,
        top_finding     TEXT,
        top_confidence  REAL,
        positive_count  INTEGER,
        all_findings    TEXT,
        doctor_notes    TEXT,
        processed_at    TEXT
    );
    """)

    # Seed default admin doctor
    existing = cur.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
    if existing == 0:
        cur.execute("""
            INSERT INTO doctors (username, password_hash, full_name, specialty, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "admin",
            hashlib.sha256("admin123".encode()).hexdigest(),
            "Dr. Admin", "General Practice",
            datetime.now().isoformat()
        ))
        con.commit()
        print("Default doctor created: username=admin, password=admin123")

    con.commit()

    # ── Migrate existing tables ─────────────────────────────────────────────
    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(patients)")}
    for col, col_type in {
        "date_of_birth": "TEXT", "phone": "TEXT", "email": "TEXT",
        "emergency_contact": "TEXT", "allergies": "TEXT", "created_at": "TEXT",
    }.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE patients ADD COLUMN {col} {col_type}")

    visit_cols = {row[1] for row in cur.execute("PRAGMA table_info(visits)")}
    for col, col_type in {
        "doctor_id": "INTEGER", "doctor_notes": "TEXT", "audio_duration_sec": "REAL",
    }.items():
        if col not in visit_cols:
            cur.execute(f"ALTER TABLE visits ADD COLUMN {col} {col_type}")

    con.commit()
    con.close()

init_db()

# ── Auth helpers ────────────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def verify_login(username, password):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT doctor_id, full_name, specialty FROM doctors WHERE username=? AND password_hash=?",
        (username, hash_password(password))
    ).fetchone()
    con.close()
    return row

def register_doctor(username, password, full_name, specialty):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("""
            INSERT INTO doctors (username, password_hash, full_name, specialty, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (username, hash_password(password), full_name, specialty, datetime.now().isoformat()))
        con.commit()
        con.close()
        return True
    except sqlite3.IntegrityError:
        return False

# ── Patient helpers ─────────────────────────────────────────────────────────────
def generate_patient_id(name, dob):
    raw = f"{name.lower().strip()}{dob}"
    return "PAT-" + hashlib.md5(raw.encode()).hexdigest()[:8].upper()

def get_or_create_patient(data):
    pid = generate_patient_id(data["full_name"], data["date_of_birth"])
    con = sqlite3.connect(DB_PATH)
    existing     = con.execute("SELECT patient_id FROM patients WHERE patient_id=?", (pid,)).fetchone()
    is_returning = existing is not None
    if not existing:
        con.execute("""
            INSERT INTO patients
                (patient_id, full_name, date_of_birth, age, gender, blood_type,
                 phone, email, insurance_status, emergency_contact, allergies, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pid, data["full_name"], data["date_of_birth"], data["age"],
            data["gender"], data["blood_type"], data["phone"], data["email"],
            data["insurance_status"], data["emergency_contact"], data["allergies"],
            datetime.now().isoformat()
        ))
        con.commit()
    con.close()
    return pid, is_returning

def save_visit(patient_id, doctor_id, visit_data, diagnosis_list):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO visits
            (patient_id, doctor_id, visit_datetime, visit_type, department,
             chief_complaint, asr_transcription, symptom_top1, symptom_top1_score,
             symptom_top2, symptom_top2_score, symptom_top3, symptom_top3_score,
             urgency, severity_score, follow_up_needed, doctor_notes,
             recommended_steps, audio_duration_sec, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id, doctor_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        visit_data.get("visit_type"), visit_data.get("department"),
        visit_data.get("chief_complaint"), visit_data.get("transcription"),
        visit_data.get("symptom_top1"), visit_data.get("symptom_top1_score"),
        visit_data.get("symptom_top2"), visit_data.get("symptom_top2_score"),
        visit_data.get("symptom_top3"), visit_data.get("symptom_top3_score"),
        visit_data.get("urgency"), visit_data.get("severity_score"),
        int(visit_data.get("follow_up_needed", False)),
        visit_data.get("doctor_notes"), visit_data.get("recommended_steps"),
        visit_data.get("audio_duration_sec"), datetime.now().isoformat()
    ))
    visit_id = cur.lastrowid
    for rank, diag in enumerate(diagnosis_list, 1):
        cur.execute("""
            INSERT INTO diagnoses (visit_id, rank, disease, likelihood, reasoning)
            VALUES (?, ?, ?, ?, ?)
        """, (visit_id, rank, diag.get("disease"), diag.get("likelihood"), diag.get("reasoning")))
    con.commit()
    con.close()
    return visit_id

def get_patient_history(patient_id):
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT v.visit_id, v.visit_datetime, v.visit_type, v.department,
               v.chief_complaint, v.asr_transcription,
               v.symptom_top1, v.symptom_top1_score,
               v.symptom_top2, v.symptom_top2_score,
               v.symptom_top3, v.symptom_top3_score,
               v.urgency, v.severity_score, v.follow_up_needed,
               v.doctor_notes, v.recommended_steps, v.audio_duration_sec,
               v.processed_at,
               d.full_name AS doctor_name,
               d.specialty AS doctor_specialty
        FROM visits v
        LEFT JOIN doctors d ON v.doctor_id = d.doctor_id
        WHERE v.patient_id = ?
        ORDER BY v.visit_datetime DESC
    """, con, params=(patient_id,))
    con.close()
    return df

def get_all_visits():
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT v.visit_id, v.patient_id, v.visit_datetime, v.visit_type, v.department,
               v.chief_complaint, v.asr_transcription,
               v.symptom_top1, v.symptom_top1_score,
               v.symptom_top2, v.symptom_top2_score,
               v.symptom_top3, v.symptom_top3_score,
               v.urgency, v.severity_score, v.follow_up_needed,
               v.doctor_notes, v.recommended_steps, v.audio_duration_sec,
               v.processed_at,
               p.full_name AS full_name,
               p.age, p.gender, p.blood_type, p.insurance_status,
               d.full_name AS doctor_name
        FROM visits v
        JOIN patients p ON v.patient_id = p.patient_id
        LEFT JOIN doctors d ON v.doctor_id = d.doctor_id
    """, con)
    con.close()
    return df

# ── X-Ray DB helpers ────────────────────────────────────────────────────────────
def save_xray_scan(patient_id, patient_name, doctor_id,
                   age, gender, view, probs_dict, top_finding,
                   top_conf, positive_count, notes=""):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO xray_scans
            (patient_id, patient_name, doctor_id, scan_datetime,
             patient_age, patient_gender, view_position,
             top_finding, top_confidence, positive_count,
             all_findings, doctor_notes, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id, patient_name, doctor_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        age, gender, view,
        top_finding, top_conf, positive_count,
        json.dumps(probs_dict),
        notes, datetime.now().isoformat()
    ))
    con.commit()
    con.close()

def get_all_xray_scans():
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT x.scan_id, x.scan_datetime, x.patient_name,
               x.patient_age, x.patient_gender, x.view_position,
               x.top_finding, x.top_confidence, x.positive_count,
               x.all_findings, x.doctor_notes,
               d.full_name AS doctor_name
        FROM xray_scans x
        LEFT JOIN doctors d ON x.doctor_id = d.doctor_id
        ORDER BY x.scan_datetime DESC
    """, con)
    con.close()
    return df

# ── Whisper ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model("small", download_root=WHISPER_CACHE)

whisper_model = load_whisper()

def audio_to_text(audio_file):
    audio_bytes = audio_file.read()
    data, sr    = sf.read(io.BytesIO(audio_bytes))
    data        = np.asarray(data, dtype=np.float32)
    result      = whisper_model.transcribe(data, fp16=False)
    duration    = len(data) / sr
    return result["text"], duration

# ── Vision model loader ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading X-Ray AI engine…")
def load_xray_engine():
    vision = load_vision_engine()
    cam    = GradCAM(vision)
    return vision, cam

vision_eng, gradcam_eng = load_xray_engine()

# ── Session state init ──────────────────────────────────────────────────────────
if "doctor" not in st.session_state:
    st.session_state.doctor = None

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN / REGISTER SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.doctor is None:
    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🏥 Seha Track Pro")
        st.markdown("#### Clinical Intelligence Platform")
        st.markdown("---")

        tab_login, tab_register = st.tabs(["🔐 Doctor Login", "➕ Register"])

        with tab_login:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pw")
            if st.button("Log In", key="btn_login"):
                result = verify_login(username, password)
                if result:
                    st.session_state.doctor = {
                        "id": result[0], "name": result[1],
                        "specialty": result[2], "username": username
                    }
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_register:
            st.caption("Register a new doctor account")
            r_name = st.text_input("Full name (e.g. Dr. Sara Ali)")
            r_spec = st.selectbox("Specialty", [
                "General Practice", "Internal Medicine", "Pulmonology",
                "Cardiology", "Neurology", "Gastroenterology",
                "Orthopaedics", "Emergency Medicine", "Paediatrics", "Other"
            ])
            r_user = st.text_input("Username")
            r_pw   = st.text_input("Password", type="password")
            r_pw2  = st.text_input("Confirm password", type="password")
            if st.button("Create Account", key="btn_register"):
                if r_pw != r_pw2:
                    st.error("Passwords do not match.")
                elif len(r_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif not r_user or not r_name:
                    st.error("Please fill in all fields.")
                else:
                    ok = register_doctor(r_user, r_pw, r_name, r_spec)
                    if ok:
                        st.success("Account created! You can now log in.")
                    else:
                        st.error("Username already taken.")
    st.stop()

doc = st.session_state.doctor

with st.sidebar:
    st.markdown(f"<div class='doctor-badge'>👨‍⚕️ {doc['name']}</div>", unsafe_allow_html=True)
    st.caption(doc["specialty"])
    st.markdown("---")
    page = st.radio("MENU", [
        "🩺 Diagnostic Lab",
        "📊 Insights Hub",
        "📋 Clinical Logs",
        "👤 Patient Search",
        "🫁 X-Ray Dashboard",
        "🫁 X-Ray Logs",
    ])
    st.markdown("---")
    if st.button("🚪 Log Out"):
        st.session_state.doctor = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Diagnostic Lab
# ══════════════════════════════════════════════════════════════════════════════
if page == "🩺 Diagnostic Lab":
    st.header("🩺 Diagnostic Lab")
    st.caption(f"Logged in as **{doc['name']}** · {doc['specialty']}")

    # ── Step 1: Patient info ──────────────────────────────────────────────────
    st.markdown("### 1 — Patient Information")
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            p_name  = st.text_input("Full name *", placeholder="e.g. Ahmed Khalil")
            p_dob   = st.date_input("Date of birth *", min_value=datetime(1900, 1, 1).date())
            p_phone = st.text_input("Phone number", placeholder="+20 1xx xxx xxxx")
        with c2:
            p_gender    = st.selectbox("Gender *", ["Male", "Female", "Other"])
            p_blood     = st.selectbox("Blood type", ["Unknown", "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
            p_insurance = st.selectbox("Insurance status", ["Insured", "Uninsured", "Government"])
        with c3:
            p_email     = st.text_input("Email", placeholder="patient@email.com")
            p_emergency = st.text_input("Emergency contact", placeholder="Name — phone")
            p_allergies = st.text_area("Known allergies", placeholder="e.g. Penicillin, Aspirin", height=68)

        age = int((datetime.now().date() - p_dob).days / 365.25)

    # ── Step 2: Visit details ─────────────────────────────────────────────────
    st.markdown("### 2 — Visit Details")
    v1, v2, v3 = st.columns(3)
    with v1:
        visit_type = st.selectbox("Visit type", ["Walk-in", "Scheduled", "Emergency", "Telehealth"])
    with v2:
        department = st.selectbox("Department", [
            "General Practice", "Internal Medicine", "Pulmonology",
            "Cardiology", "Neurology", "Gastroenterology",
            "Orthopaedics", "Emergency Medicine", "Paediatrics"
        ])
    with v3:
        severity = st.slider("Severity (1-10)", 1, 10, 5)

    follow_up    = st.checkbox("Follow-up required")
    doctor_notes = st.text_area("Doctor notes", placeholder="Additional clinical observations...", height=80)

    # ── Step 3: Symptom input ─────────────────────────────────────────────────
    st.markdown("### 3 — Symptom Input")
    audio_input = st.audio_input("🎙️ Record symptoms (optional)")
    user_text   = st.text_input("Or type symptoms manually",
                                placeholder="e.g. persistent cough and fever for 3 days")

    # ── Analyze button ────────────────────────────────────────────────────────
    if st.button("🔍 Analyze & Save to Database"):
        if not p_name:
            st.error("Patient full name is required.")
            st.stop()

        with st.spinner("Running AI pipeline..."):
            transcription  = ""
            audio_duration = None

            if audio_input:
                try:
                    transcription, audio_duration = audio_to_text(audio_input)
                    st.info(f"🎙️ Transcribed: *{transcription}*")
                    text_to_analyze = transcription
                except Exception as e:
                    st.warning(f"Audio transcription failed: {e}")
                    text_to_analyze = user_text or "fever and cough"
            else:
                text_to_analyze = user_text or "fever and cough"
                transcription   = text_to_analyze

            try:
                predictions = predict_topk(text_to_analyze)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.stop()

            top1 = predictions[0] if len(predictions) > 0 else {"label": "unknown", "score": 0}
            top2 = predictions[1] if len(predictions) > 1 else {"label": None, "score": None}
            top3 = predictions[2] if len(predictions) > 2 else {"label": None, "score": None}

            urgency_map = {
                range(1, 4):  "routine",
                range(4, 6):  "soon",
                range(6, 8):  "urgent",
                range(8, 11): "emergency"
            }
            urgency = next((v for k, v in urgency_map.items() if severity in k), "routine")

            patient_data = {
                "full_name":         p_name,
                "date_of_birth":     str(p_dob),
                "age":               age,
                "gender":            p_gender,
                "blood_type":        p_blood,
                "phone":             p_phone,
                "email":             p_email,
                "insurance_status":  p_insurance,
                "emergency_contact": p_emergency,
                "allergies":         p_allergies,
            }
            patient_id, is_returning = get_or_create_patient(patient_data)

            visit_data = {
                "visit_type":         visit_type,
                "department":         department,
                "chief_complaint":    text_to_analyze,
                "transcription":      transcription,
                "symptom_top1":       top1["label"],
                "symptom_top1_score": top1["score"],
                "symptom_top2":       top2["label"],
                "symptom_top2_score": top2["score"],
                "symptom_top3":       top3["label"],
                "symptom_top3_score": top3["score"],
                "urgency":            urgency,
                "severity_score":     severity,
                "follow_up_needed":   follow_up,
                "doctor_notes":       doctor_notes,
                "recommended_steps":  "",
                "audio_duration_sec": audio_duration,
            }
            visit_id = save_visit(patient_id, doc["id"], visit_data, [])
            # Store patient_id in session for Step 4
            st.session_state["last_patient_id"]   = patient_id
            st.session_state["last_patient_name"] = p_name

        if is_returning:
            hist_count = len(get_patient_history(patient_id))
            st.info(f"🔁 Returning patient — {hist_count} previous visit(s) on record. New visit added.")
        else:
            st.success("✅ New patient registered and visit saved.")

        st.caption(f"Patient ID: `{patient_id}` · Visit ID: `{visit_id}`")

        if transcription:
            st.markdown("#### 🎙️ Transcription")
            st.info(f'"{transcription}"')
            if audio_duration:
                st.caption(f"Audio duration: {audio_duration:.1f}s")

        URGENCY_COLOR = {"routine": "🟢", "soon": "🟡", "urgent": "🟠", "emergency": "🔴"}
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Top Symptom", top1["label"])
        r2.metric("Confidence",  f"{top1['score']:.1%}")
        r3.metric("Severity",    f"{severity}/10")
        r4.metric("Urgency",     f"{URGENCY_COLOR.get(urgency,'')} {urgency.capitalize()}")

        st.markdown("#### 🩺 All Predicted Symptoms")
        pred_df         = pd.DataFrame(predictions)
        pred_df["pct"]  = (pred_df["score"] * 100).round(2)
        colors          = ["#2563eb" if i == 0 else "#93c5fd" for i in range(len(pred_df))]

        fig_preds = go.Figure(go.Bar(
            x=pred_df["pct"],
            y=pred_df["label"],
            orientation="h",
            marker_color=colors,
            text=pred_df["pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig_preds.update_layout(
            xaxis=dict(title="Confidence (%)", range=[0, max(pred_df["pct"]) * 1.15]),
            yaxis=dict(autorange="reversed"),
            height=max(300, len(pred_df) * 32),
            margin=dict(l=10, r=40, t=10, b=30),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_preds, use_container_width=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=top1["score"] * 100,
            title={"text": f"Confidence — {top1['label']}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2563eb"},
                "steps": [
                    {"range": [0,  50],  "color": "#fee2e2"},
                    {"range": [50, 75],  "color": "#fef9c3"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"#### 📋 Complete medical history — {p_name}")
        hist = get_patient_history(patient_id)
        if not hist.empty:
            display_hist = hist[[
                "visit_datetime", "visit_type", "department",
                "chief_complaint", "asr_transcription",
                "symptom_top1", "symptom_top1_score",
                "symptom_top2", "symptom_top3",
                "urgency", "severity_score", "follow_up_needed",
                "doctor_name", "doctor_notes"
            ]].copy()
            display_hist.columns = [
                "Date", "Type", "Department",
                "Chief Complaint", "Transcription",
                "Symptom 1", "Confidence",
                "Symptom 2", "Symptom 3",
                "Urgency", "Severity", "Follow-up",
                "Doctor", "Notes"
            ]
            display_hist["Confidence"] = display_hist["Confidence"].apply(
                lambda v: f"{v:.1%}" if pd.notna(v) else ""
            )
            st.dataframe(display_hist, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ── Step 4: X-Ray Analysis (independent — always visible) ────────────────
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown("### 4 — Chest X-Ray Analysis")
    st.caption("Optional: upload a frontal X-ray to run the DenseNet121 multimodal model.")

    with st.container():
        # ── Patient metadata for the X-ray (entered directly here) ───────────
        # Auto-fill name from Step 1/3 if already entered, but always editable
        _default_name = st.session_state.get("last_patient_name", p_name or "")
        xray_patient_name = st.text_input(
            "Patient name *",
            value=_default_name,
            placeholder="e.g. Ahmed Khalil",
            key="xray_patient_name_input",
            help="Auto-filled if Step 1 was completed — edit freely for standalone scans"
        )

        xr1, xr2, xr3, xr4 = st.columns(4)
        with xr1:
            xray_age    = st.number_input("Patient age *", min_value=1, max_value=100,
                                          value=40, step=1, key="xray_age_input",
                                          help="Enter the actual patient age for this scan")
        with xr2:
            xray_gender = st.selectbox("Gender *", ["Male", "Female"],
                                       key="xray_gender_input",
                                       help="Required — affects model prediction")
        with xr3:
            xray_view   = st.selectbox("X-Ray view *", ["PA", "AP"],
                                       key="xray_view",
                                       help="PA = Posteroanterior · AP = Anteroposterior")
        with xr4:
            use_custom_t = st.toggle("Override thresholds", value=False,
                                     key="xray_thresh_toggle")

        global_t   = st.slider("Global threshold", 0.05, 0.95, 0.35, 0.05,
                               disabled=not use_custom_t, key="xray_global_t")
        xray_notes = st.text_input("X-Ray notes (optional)",
                                   placeholder="e.g. portable AP, post-op day 2")
        xray_file  = st.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"],
                                      key="xray_uploader")

    if xray_file:
        col_orig, col_res = st.columns([1, 1.6], gap="large")

        with col_orig:
            pil_img = Image.open(xray_file).convert("RGB")
            st.image(pil_img, caption="Uploaded X-Ray", use_container_width=True)

        if st.button("🫁 Run X-Ray Analysis", key="btn_xray"):
            with st.spinner("Preprocessing with CLAHE…"):
                clahe_img   = apply_clahe(pil_img)
                img_tensor  = VAL_TRANSFORM(clahe_img)
                meta_tensor = encode_meta(xray_age, xray_gender, xray_view)

            with st.spinner("Running DenseNet121 inference…"):
                with torch.no_grad():
                    logits = vision_eng(img_tensor.unsqueeze(0).to(DEVICE), meta_tensor)
                    probs  = torch.sigmoid(logits)[0].cpu().numpy()

            thresholds = (
                [global_t] * 14
                if use_custom_t
                else [OPTIMAL_THRESHOLDS[lbl] for lbl in DISEASE_LABELS]
            )
            positives  = [DISEASE_LABELS[i] for i in range(14) if probs[i] >= thresholds[i]]
            top_idx    = int(np.argmax(probs))
            top_name   = DISEASE_LABELS[top_idx]
            top_conf   = float(probs[top_idx])

            with col_res:
                # ── Metrics ────────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("Findings",      len(positives))
                m2.metric("Top finding",   top_name)
                m3.metric("Confidence",    f"{top_conf:.1%}")

                # ── Banner ─────────────────────────────────────────────────
                if positives:
                    pos_html = " ".join(f'<span class="xray-badge">{d}</span>' for d in positives)
                    st.markdown(
                        f'<div class="xray-positive">⚠️ Pathology detected<br>{pos_html}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="xray-negative">✅ No significant pathology above threshold</div>',
                        unsafe_allow_html=True
                    )

                # ── Grad-CAM ───────────────────────────────────────────────
                st.markdown("**Grad-CAM — Model Attention**")
                try:
                    cam_np      = gradcam_eng.generate(img_tensor, meta_tensor, top_idx)
                    overlay_img = GradCAM.overlay(pil_img, cam_np, alpha=0.45)
                    st.image(overlay_img,
                             caption=f"Attention map for: {top_name}",
                             use_container_width=True)
                    st.caption("Red/yellow = regions that most influenced the prediction.")
                except Exception as e:
                    st.warning(f"Grad-CAM error: {e}")

                # ── Probability bar chart ──────────────────────────────────
                st.markdown("**All 14 Disease Probabilities**")
                sorted_i  = np.argsort(probs)[::-1]
                prob_df   = pd.DataFrame({
                    "Disease":     [DISEASE_LABELS[i] for i in sorted_i],
                    "Probability": [round(float(probs[i]) * 100, 2) for i in sorted_i],
                    "Detected":    ["Yes" if probs[i] >= thresholds[i] else "No"
                                    for i in sorted_i],
                })
                bar_colors = ["#ef4444" if r == "Yes" else "#93c5fd"
                              for r in prob_df["Detected"]]
                fig_xray = go.Figure(go.Bar(
                    x=prob_df["Probability"],
                    y=prob_df["Disease"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=prob_df["Probability"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                ))
                fig_xray.update_layout(
                    xaxis=dict(title="Probability (%)", range=[0, max(prob_df["Probability"]) * 1.2]),
                    yaxis=dict(autorange="reversed"),
                    height=480,
                    margin=dict(l=10, r=50, t=10, b=30),
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig_xray, use_container_width=True)

            # ── Save scan to DB ────────────────────────────────────────────
            pid_for_scan  = st.session_state.get("last_patient_id", None)
            # Always use the explicitly entered name — never guess
            name_for_scan = xray_patient_name.strip() or "Unknown" 
            probs_dict    = {DISEASE_LABELS[i]: round(float(probs[i]), 4)
                             for i in range(14)}
            save_xray_scan(
                patient_id=pid_for_scan, patient_name=name_for_scan,
                doctor_id=doc["id"],
                age=xray_age, gender=xray_gender, view=xray_view,
                probs_dict=probs_dict,
                top_finding=top_name, top_conf=top_conf,
                positive_count=len(positives),
                notes=xray_notes,
            )
            st.success("✅ X-ray scan saved to Clinical Logs.")

            # ── CSV download ───────────────────────────────────────────────
            csv = prob_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download results CSV", csv,
                               "xray_results.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Insights Hub  (original — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Insights Hub":
    st.title("📊 Clinical Intelligence Dashboard")

    df = get_all_visits()
    if df.empty:
        st.warning("No visit data yet. Run some diagnoses first.")
        st.stop()

    df["visit_datetime"] = pd.to_datetime(df["visit_datetime"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients",    df["patient_id"].nunique())
    m2.metric("Total Visits",      len(df))
    m3.metric("Avg Severity",      f"{df['severity_score'].mean():.1f}")
    m4.metric("Follow-ups Needed", int(df["follow_up_needed"].sum()))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Urgency Distribution")
        fig_u = px.pie(df, names="urgency", hole=0.5,
                       color="urgency",
                       color_discrete_map={"routine": "#22C55E", "soon": "#F59E0B",
                                           "urgent": "#EF4444", "emergency": "#7C3AED"})
        st.plotly_chart(fig_u, use_container_width=True)
    with c2:
        st.markdown("#### Top Predicted Symptoms")
        s_cnt = df["symptom_top1"].value_counts().head(10).reset_index()
        s_cnt.columns = ["symptom", "count"]
        fig_s = px.bar(s_cnt.sort_values("count"), x="count", y="symptom",
                       orientation="h", color="count", color_continuous_scale="Blues")
        fig_s.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("#### Weekly Visit Volume")
    daily         = df.set_index("visit_datetime").resample("W")["visit_id"].count().reset_index()
    daily.columns = ["week", "visits"]
    fig_v = px.area(daily, x="week", y="visits", color_discrete_sequence=["#3b82f6"])
    st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("#### Patient Risk Matrix — Age vs Severity")
    fig_r = px.scatter(df, x="age", y="severity_score",
                       color="urgency", size="severity_score",
                       hover_name="full_name",
                       color_discrete_map={"routine": "#22C55E", "soon": "#F59E0B",
                                           "urgent": "#EF4444", "emergency": "#7C3AED"},
                       template="plotly_white")
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("#### Severity Distribution per Symptom")
    top_symp = df["symptom_top1"].value_counts().head(8).index
    fig_b = px.box(df[df["symptom_top1"].isin(top_symp)],
                   x="symptom_top1", y="severity_score", color="symptom_top1")
    fig_b.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_b, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Clinical Logs  (original — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Clinical Logs":
    st.header("📋 Clinical Logs")

    df = get_all_visits()
    if df.empty:
        st.warning("No records found.")
        st.stop()

    df["visit_datetime"] = pd.to_datetime(df["visit_datetime"])

    f1, f2, f3 = st.columns(3)
    with f1:
        urgency_filter = st.multiselect("Urgency", df["urgency"].dropna().unique().tolist(),
                                        default=df["urgency"].dropna().unique().tolist())
    with f2:
        dept_filter = st.multiselect("Department", df["department"].dropna().unique().tolist(),
                                     default=df["department"].dropna().unique().tolist())
    with f3:
        date_range = st.date_input("Date range",
                                   [df["visit_datetime"].min().date(),
                                    df["visit_datetime"].max().date()])

    filtered = df[
        df["urgency"].isin(urgency_filter) &
        df["department"].isin(dept_filter) &
        (df["visit_datetime"].dt.date >= date_range[0]) &
        (df["visit_datetime"].dt.date <= date_range[1])
    ] if len(date_range) == 2 else df

    st.caption(f"Showing {len(filtered)} records")
    st.dataframe(
        filtered[["visit_datetime", "full_name", "age", "gender",
                  "chief_complaint", "symptom_top1", "urgency",
                  "severity_score", "follow_up_needed", "doctor_name", "department"]],
        use_container_width=True
    )
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export CSV", csv, "clinical_logs.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Patient Search  (original — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Patient Search":
    st.header("👤 Patient Search")
    search = st.text_input("Search by patient name", placeholder="e.g. Ahmed")

    if search:
        con      = sqlite3.connect(DB_PATH)
        patients = pd.read_sql_query(
            "SELECT * FROM patients WHERE full_name LIKE ?",
            con, params=(f"%{search}%",)
        )
        con.close()

        if patients.empty:
            st.info("No patients found.")
        else:
            for _, p in patients.iterrows():
                with st.expander(f"🧑 {p['full_name']} — {p['gender']}, age {p['age']} | {p['blood_type']} | {p['insurance_status']}"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.write(f"**Patient ID:** {p['patient_id']}")
                        st.write(f"**DOB:** {p['date_of_birth']}")
                        st.write(f"**Phone:** {p['phone']}")
                        st.write(f"**Email:** {p['email']}")
                    with d2:
                        st.write(f"**Emergency contact:** {p['emergency_contact']}")
                        st.write(f"**Allergies:** {p['allergies']}")
                        st.write(f"**Registered:** {p['created_at']}")

                    st.markdown("**Visit history:**")
                    hist = get_patient_history(p["patient_id"])
                    if hist.empty:
                        st.caption("No visits recorded yet.")
                    else:
                        st.dataframe(
                            hist[["visit_datetime", "chief_complaint", "symptom_top1",
                                  "urgency", "severity_score", "doctor_name", "doctor_notes"]],
                            use_container_width=True
                        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — X-Ray Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🫁 X-Ray Dashboard":
    st.title("🫁 X-Ray Intelligence Dashboard")

    df = get_all_xray_scans()
    if df.empty:
        st.warning("No X-ray scans recorded yet. Run an analysis in the Diagnostic Lab first.")
        st.stop()

    df["scan_datetime"] = pd.to_datetime(df["scan_datetime"])

    # ── Top metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Scans",        len(df))
    m2.metric("Unique Patients",    df["patient_name"].nunique())
    m3.metric("Avg Findings/Scan",  f"{df['positive_count'].mean():.1f}")
    m4.metric("Scans with Findings",int((df["positive_count"] > 0).sum()))

    st.markdown("---")
    c1, c2 = st.columns(2)

    # ── Top findings distribution ──────────────────────────────────────────
    with c1:
        st.markdown("#### Most Frequent Top Finding")
        top_counts = df["top_finding"].value_counts().reset_index()
        top_counts.columns = ["Disease", "Count"]
        fig_top = px.bar(top_counts.sort_values("Count"),
                         x="Count", y="Disease", orientation="h",
                         color="Count", color_continuous_scale="Reds")
        fig_top.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                              margin=dict(l=10, r=30, t=10, b=30))
        st.plotly_chart(fig_top, use_container_width=True)

    # ── Gender distribution of scans ──────────────────────────────────────
    with c2:
        st.markdown("#### Gender Distribution of Scans")
        fig_g = px.pie(df, names="patient_gender", hole=0.5,
                       color_discrete_sequence=["#3b82f6", "#ec4899"])
        st.plotly_chart(fig_g, use_container_width=True)

    # ── Disease frequency from all_findings JSON ──────────────────────────
    st.markdown("#### Disease Probability Heatmap (all scans)")
    try:
        rows = []
        for _, row in df.iterrows():
            if pd.notna(row["all_findings"]) and row["all_findings"]:
                findings = json.loads(row["all_findings"])
                # Use scan_id + name as label so duplicate patient names
                # never cause a "duplicate column names" crash
                label = f"#{row['scan_id']}  {(row['patient_name'] or 'Unknown').strip()}"
                findings["_label"] = label
                rows.append(findings)
        if rows:
            heat_df = pd.DataFrame(rows).set_index("_label")[DISEASE_LABELS]
            fig_heat = px.imshow(
                heat_df.T,
                color_continuous_scale="RdYlGn_r",
                zmin=0, zmax=1,
                labels=dict(x="Scan", y="Disease", color="Probability"),
                aspect="auto",
            )
            fig_heat.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=30))
            st.plotly_chart(fig_heat, use_container_width=True)
    except Exception as e:
        st.caption(f"Heatmap unavailable: {e}")

    # ── Weekly scan volume ─────────────────────────────────────────────────
    st.markdown("#### Weekly X-Ray Scan Volume")
    weekly         = df.set_index("scan_datetime").resample("W")["scan_id"].count().reset_index()
    weekly.columns = ["week", "scans"]
    fig_w = px.area(weekly, x="week", y="scans", color_discrete_sequence=["#06b6d4"])
    st.plotly_chart(fig_w, use_container_width=True)

    # ── Age vs Top Confidence ──────────────────────────────────────────────
    st.markdown("#### Patient Age vs Top Prediction Confidence")
    fig_sc = px.scatter(df, x="patient_age", y="top_confidence",
                        color="top_finding", size="positive_count",
                        hover_name="patient_name",
                        template="plotly_white",
                        labels={"top_confidence": "Confidence", "patient_age": "Age"})
    fig_sc.update_layout(height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── View position split ────────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### View Position (PA vs AP)")
        fig_v = px.pie(df, names="view_position", hole=0.5,
                       color_discrete_sequence=["#8b5cf6", "#f59e0b"])
        st.plotly_chart(fig_v, use_container_width=True)
    with c4:
        st.markdown("#### Positive Finding Count Distribution")
        fig_pos = px.histogram(df, x="positive_count", nbins=15,
                               color_discrete_sequence=["#ef4444"],
                               labels={"positive_count": "# Findings Detected"})
        fig_pos.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_pos, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — X-Ray Logs
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🫁 X-Ray Logs":
    st.header("🫁 X-Ray Scan Logs")

    df = get_all_xray_scans()
    if df.empty:
        st.warning("No X-ray scans recorded yet.")
        st.stop()

    df["scan_datetime"] = pd.to_datetime(df["scan_datetime"])

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        finding_filter = st.multiselect(
            "Top Finding",
            df["top_finding"].dropna().unique().tolist(),
            default=df["top_finding"].dropna().unique().tolist()
        )
    with f2:
        view_filter = st.multiselect(
            "View Position",
            df["view_position"].dropna().unique().tolist(),
            default=df["view_position"].dropna().unique().tolist()
        )
    with f3:
        xray_date_range = st.date_input(
            "Date range",
            [df["scan_datetime"].min().date(), df["scan_datetime"].max().date()],
            key="xray_log_dates"
        )

    filtered = df[
        df["top_finding"].isin(finding_filter) &
        df["view_position"].isin(view_filter) &
        (df["scan_datetime"].dt.date >= xray_date_range[0]) &
        (df["scan_datetime"].dt.date <= xray_date_range[1])
    ] if len(xray_date_range) == 2 else df

    st.caption(f"Showing {len(filtered)} of {len(df)} records")

    # ── Main table ────────────────────────────────────────────────────────────
    display_df = filtered[[
        "scan_datetime", "patient_name", "patient_age", "patient_gender",
        "view_position", "top_finding", "top_confidence", "positive_count",
        "doctor_name", "doctor_notes"
    ]].copy()
    display_df.columns = [
        "Date", "Patient", "Age", "Gender",
        "View", "Top Finding", "Confidence", "# Findings",
        "Doctor", "Notes"
    ]
    display_df["Confidence"] = display_df["Confidence"].apply(
        lambda v: f"{v:.1%}" if pd.notna(v) else ""
    )
    st.dataframe(display_df, use_container_width=True)

    # ── Expandable per-scan detail ────────────────────────────────────────────
    st.markdown("#### Detailed Scan View")
    for _, row in filtered.head(20).iterrows():
        with st.expander(
            f"🫁 {row['patient_name']}  ·  {row['scan_datetime'].strftime('%Y-%m-%d %H:%M')}  "
            f"·  Top: {row['top_finding']} ({row['top_confidence']:.1%})"
        ):
            d1, d2 = st.columns(2)
            with d1:
                st.write(f"**Age:** {row['patient_age']}  |  "
                         f"**Gender:** {row['patient_gender']}  |  "
                         f"**View:** {row['view_position']}")
                st.write(f"**Doctor:** {row['doctor_name'] or '—'}")
                if row["doctor_notes"]:
                    st.write(f"**Notes:** {row['doctor_notes']}")
            with d2:
                if pd.notna(row["all_findings"]) and row["all_findings"]:
                    try:
                        findings  = json.loads(row["all_findings"])
                        thresholds = [OPTIMAL_THRESHOLDS[lbl] for lbl in DISEASE_LABELS]
                        mini_df   = pd.DataFrame({
                            "Disease":     DISEASE_LABELS,
                            "Probability": [findings.get(lbl, 0) for lbl in DISEASE_LABELS],
                        }).sort_values("Probability", ascending=False)
                        mini_df["Detected"] = [
                            "✅" if findings.get(lbl, 0) >= OPTIMAL_THRESHOLDS[lbl] else "—"
                            for lbl in mini_df["Disease"]
                        ]
                        mini_df["Probability"] = mini_df["Probability"].apply(lambda v: f"{v:.1%}")
                        st.dataframe(mini_df, use_container_width=True, hide_index=True)
                    except Exception:
                        st.caption("Findings data unavailable.")

    # ── Export ────────────────────────────────────────────────────────────────
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export X-Ray Logs CSV", csv, "xray_logs.csv", "text/csv")
