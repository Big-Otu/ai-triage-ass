import streamlit as st
import joblib
import os
import pandas as pd
from datetime import datetime
from preprocess import clean_symptom_text, URGENCY_COLORS, URGENCY_ADVICE
from security import (
    authenticate_user, create_user, log_audit, load_audit_log,
    sanitize_input, validate_username, validate_password_strength,
    encrypt_record, decrypt_record, is_session_expired,
    get_remaining_session_time, SESSION_TIMEOUT_MINUTES, ENC_HISTORY
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Triage Assistant", page_icon="🏥", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header { background: linear-gradient(135deg, #1a5276, #2e86c1); padding: 2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 2rem; }
.login-card  { background: white; padding: 2.5rem; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.12); max-width: 420px; margin: 3rem auto; }
.result-card { padding: 1.5rem; border-radius: 12px; border-left: 6px solid; margin: 1rem 0; }
.result-high   { border-color: #e74c3c; background: #fdf2f2; }
.result-medium { border-color: #f39c12; background: #fef9f0; }
.result-low    { border-color: #27ae60; background: #f0faf4; }
.metric-box  { background: white; border-radius: 10px; padding: 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.stat-box    { background: white; border-radius: 10px; padding: 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid; }
.symptom-tag { display:inline-block; background:#eaf2ff; color:#2980b9; padding:4px 12px; border-radius:20px; margin:3px; font-size:0.85rem; }
.disclaimer  { background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:0.8rem 1rem; font-size:0.85rem; color:#856404; margin-top:1rem; }
.sec-badge   { background:#1a5276; color:white; padding:3px 10px; border-radius:20px; font-size:0.75rem; margin:2px; display:inline-block; }
.timeout-bar { background:#fff3cd; border:1px solid #f39c12; border-radius:8px; padding:0.5rem 1rem; font-size:0.85rem; color:#856404; margin-bottom:1rem; }
.audit-row-login  { background:#f0faf4; }
.audit-row-blocked{ background:#fdf2f2; }
</style>
""", unsafe_allow_html=True)

COMMON_SYMPTOMS = [
    "fever","headache","vomiting","nausea","chills","fatigue","sweating","body ache",
    "diarrhea","cough","sore throat","chest pain","difficulty breathing","skin rash",
    "itching","abdominal pain","loss of appetite","dizziness","yellow eyes","joint pain",
    "runny nose","back pain","weakness","weight loss","blurred vision","high temperature",
    "swollen lymph nodes","stiff neck"
]

# ─── Patient History (encrypted) ──────────────────────────────────────────────
def save_to_history(username, name, age, sex, symptoms, duration, urgency, condition):
    record = {
        "Timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "Name": name or "Unknown", "Age": age, "Sex": sex,
        "Symptoms": symptoms, "Duration": duration,
        "Urgency": urgency, "Condition": condition,
        "AssessedBy": username
    }
    encrypted = encrypt_record(record)
    df_new = pd.DataFrame([encrypted])
    if os.path.exists(ENC_HISTORY):
        df_combined = pd.concat([pd.read_csv(ENC_HISTORY), df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(ENC_HISTORY, index=False)
    log_audit(username, "PATIENT_ASSESSED", f"Urgency:{urgency} Condition:{condition}")

def load_history():
    if not os.path.exists(ENC_HISTORY):
        return pd.DataFrame()
    df = pd.read_csv(ENC_HISTORY)
    if df.empty:
        return df
    decrypted_rows = [decrypt_record(row) for row in df.to_dict("records")]
    return pd.DataFrame(decrypted_rows)

# ─── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return joblib.load("model/triage_model.pkl"), joblib.load("model/disease_model.pkl")

def run_prediction(symptom_text):
    um, dm = load_models()
    cleaned = clean_symptom_text(symptom_text)
    urgency = um.predict([cleaned])[0]
    disease = dm.predict([cleaned])[0]
    proba   = um.predict_proba([cleaned])[0]
    return urgency, disease.title(), dict(zip(um.classes_, proba))

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

# ══════════════════════════════════════════════════════════════════════════════
# SESSION TIMEOUT CHECK
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.authenticated:
    if is_session_expired(st.session_state.last_activity):
        log_audit(st.session_state.username, "SESSION_TIMEOUT", "Auto logged out")
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.warning("⏱️ Your session expired due to inactivity. Please log in again.")
    else:
        st.session_state.last_activity = datetime.now()

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown('<div class="main-header"><h1>🏥 AI Patient Triage Assistant</h1><p>Secure Healthcare AI System — Ghana Clinic Edition</p></div>', unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Secure Login")
        st.caption("Authorised personnel only")

        with st.form("login_form"):
            username_input = st.text_input("Username", placeholder="Enter username")
            password_input = st.text_input("Password", type="password", placeholder="Enter password")
            login_btn = st.form_submit_button("Login →", use_container_width=True, type="primary")

        if login_btn:
            # Validate inputs first
            u_check = sanitize_input(username_input, "Username")
            p_check = sanitize_input(password_input, "Password")

            if not u_check["valid"]:
                st.error(u_check["warning"])
            elif not p_check["valid"]:
                st.error(p_check["warning"])
            elif not username_input or not password_input:
                st.error("Please enter both username and password.")
            else:
                result = authenticate_user(username_input, password_input)
                if result["success"]:
                    st.session_state.authenticated = True
                    st.session_state.username = username_input
                    st.session_state.role = result["role"]
                    st.session_state.last_activity = datetime.now()
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")

        st.markdown("---")
        st.markdown("**Default credentials for testing:**")
        st.code("Admin:   admin   / Admin@1234\nNurse:   nurse1  / Nurse@1234\nDoctor:  doctor1 / Doctor@1234")
        st.markdown('</div>', unsafe_allow_html=True)

    # Security badges
    st.markdown("---")
    cx = st.columns(5)
    badges = ["🔐 Bcrypt Auth","🔒 AES Encryption","📝 Audit Trail","🛡️ Input Validation","⏱️ Session Timeout"]
    for i, b in enumerate(badges):
        with cx[i]:
            st.markdown(f'<div style="text-align:center"><span class="sec-badge">{b}</span></div>', unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (authenticated)
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=70)
    st.markdown(f"### 👤 {st.session_state.username}")
    st.markdown(f"**Role:** `{st.session_state.role.upper()}`")
    st.markdown("---")

    # Session timer
    remaining = get_remaining_session_time(st.session_state.last_activity)
    mins, secs = divmod(remaining, 60)
    color = "#e74c3c" if remaining < 120 else "#27ae60"
    st.markdown(f'<div style="background:#f8f9fa;border-radius:8px;padding:0.6rem;text-align:center;border-left:4px solid {color}"><span style="font-size:0.8rem;color:#777">SESSION EXPIRES IN</span><br><span style="font-size:1.3rem;font-weight:700;color:{color}">{mins:02d}:{secs:02d}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Urgency Levels**")
    st.markdown("🔴 **HIGH** — Immediate\n🟡 **MEDIUM** — Within 1hr\n🟢 **LOW** — Routine")
    st.markdown("---")

    h = load_history()
    if not h.empty:
        st.markdown("**📊 Stats**")
        st.metric("Total Patients", len(h))
        st.metric("High Priority", len(h[h['Urgency']=='HIGH']) if 'Urgency' in h.columns else 0)

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        log_audit(st.session_state.username, "LOGOUT", "User logged out")
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()
    st.caption("⚠️ AI prototype. Consult a medical professional.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>🏥 AI Patient Triage Assistant</h1><p>Helping Ghana\'s clinics prioritize patient care with Artificial Intelligence</p></div>', unsafe_allow_html=True)

# Security badges strip
badges = ["🔐 Authenticated","🔒 Data Encrypted","📝 Audit Logging","🛡️ Input Validated","⏱️ Session Managed"]
st.markdown(' '.join([f'<span class="sec-badge">{b}</span>' for b in badges]), unsafe_allow_html=True)
st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = ["🩺 Assess Patient", "📋 Patient History", "📜 Audit Log"]
if st.session_state.role == "admin":
    tabs.append("👥 User Management")

tab_objects = st.tabs(tabs)
tab_assess   = tab_objects[0]
tab_history  = tab_objects[1]
tab_audit    = tab_objects[2]
tab_admin    = tab_objects[3] if st.session_state.role == "admin" else None

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — ASSESS PATIENT
# ════════════════════════════════════════════════════════════════════════
with tab_assess:
    st.subheader("👤 Patient Information")
    c1, c2, c3 = st.columns(3)
    with c1: patient_name = st.text_input("Patient Name", placeholder="e.g. Kofi Mensah")
    with c2: patient_age  = st.number_input("Age", min_value=0, max_value=120, value=25)
    with c3: patient_sex  = st.selectbox("Sex", ["Male","Female","Other"])

    st.markdown("---")
    st.subheader("🩺 Symptom Entry")
    t1, t2 = st.tabs(["✅ Select from List", "✍️ Type Manually"])
    selected_symptoms = []

    with t1:
        cols = st.columns(4)
        for i, symptom in enumerate(COMMON_SYMPTOMS):
            with cols[i % 4]:
                if st.checkbox(symptom.capitalize(), key=f"sym_{i}"):
                    selected_symptoms.append(symptom)

    with t2:
        typed_symptoms = st.text_area("Describe symptoms:", placeholder="e.g. high fever, headache and vomiting for 3 days...", height=100)

    all_symptoms = (' '.join(selected_symptoms) + ' ' + (typed_symptoms or '')).strip()

    st.markdown("---")
    duration = st.select_slider("⏱️ Symptom duration",
        options=["< 1 hour","1–6 hours","6–24 hours","1–3 days","3–7 days","> 1 week"], value="1–3 days")

    st.markdown("---")
    cb, cc = st.columns([2,1])
    with cb: assess_clicked = st.button("🔍 Assess Patient", type="primary", use_container_width=True)
    with cc:
        if st.button("🔄 Clear", use_container_width=True): st.rerun()

    if assess_clicked:
        # ── Input validation ──
        name_check = sanitize_input(patient_name, "Patient Name")
        sym_check  = sanitize_input(all_symptoms, "Symptoms")

        if not name_check["valid"]:
            st.error(name_check["warning"])
            log_audit(st.session_state.username, "VALIDATION_FAIL", f"Name field: {name_check['warning']}")
        elif not sym_check["valid"]:
            st.error(sym_check["warning"])
            log_audit(st.session_state.username, "VALIDATION_FAIL", f"Symptom field: {sym_check['warning']}")
        elif not all_symptoms:
            st.warning("⚠️ Please enter at least one symptom.")
        elif not os.path.exists("model/triage_model.pkl"):
            st.error("❌ Model not found. Run `python model.py` first.")
        else:
            clean_name = name_check["clean_text"]
            clean_syms = sym_check["clean_text"]

            with st.spinner("Analyzing symptoms..."):
                urgency, condition, confidence = run_prediction(clean_syms or all_symptoms)

            save_to_history(st.session_state.username, clean_name, patient_age,
                patient_sex, ', '.join(selected_symptoms) if selected_symptoms else clean_syms,
                duration, urgency, condition)

            st.markdown("---")
            st.subheader("📋 Triage Result")
            badge  = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}[urgency]
            color  = URGENCY_COLORS[urgency]
            advice = URGENCY_ADVICE[urgency]

            st.markdown(f'<div class="result-card result-{urgency.lower()}"><h2 style="color:{color};margin:0">{badge} {urgency} PRIORITY</h2><p style="margin:0.5rem 0 0">{advice}</p></div>', unsafe_allow_html=True)

            ca, cb2, cc2 = st.columns(3)
            with ca:  st.markdown(f'<div class="metric-box"><div style="font-size:0.8rem;color:#777">PATIENT</div><div style="font-size:1.2rem;font-weight:600">{clean_name or "Unknown"}</div><div style="font-size:0.85rem;color:#555">{patient_age} yrs · {patient_sex}</div></div>', unsafe_allow_html=True)
            with cb2: st.markdown(f'<div class="metric-box"><div style="font-size:0.8rem;color:#777">LIKELY CONDITION</div><div style="font-size:1.2rem;font-weight:600">{condition}</div><div style="font-size:0.85rem;color:#555">AI suggestion only</div></div>', unsafe_allow_html=True)
            with cc2: st.markdown(f'<div class="metric-box"><div style="font-size:0.8rem;color:#777">DURATION</div><div style="font-size:1.2rem;font-weight:600">{duration}</div><div style="font-size:0.85rem;color:#555">As reported</div></div>', unsafe_allow_html=True)

            st.markdown("#### 📊 Model Confidence")
            conf_df = pd.DataFrame({"Urgency": list(confidence.keys()), "Confidence (%)": [round(v*100,1) for v in confidence.values()]}).sort_values("Confidence (%)", ascending=False)
            st.bar_chart(conf_df.set_index("Urgency"))

            if selected_symptoms:
                st.markdown("#### 🏷️ Symptoms Recorded")
                st.markdown(''.join([f'<span class="symptom-tag">{s}</span>' for s in selected_symptoms]), unsafe_allow_html=True)

            st.success("✅ Patient record saved (encrypted) to history log.")
            st.caption(f"🕐 {datetime.now().strftime('%d %b %Y, %I:%M %p')} | 👤 Assessed by: {st.session_state.username}")
            st.markdown('<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> AI prototype. Does not replace professional medical diagnosis.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT HISTORY
# ════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("📋 Patient History Log")
    st.caption("🔒 Records are stored encrypted and decrypted only for authorised users.")
    history_df = load_history()

    if history_df.empty:
        st.info("📭 No records yet.")
    else:
        total  = len(history_df)
        high   = len(history_df[history_df['Urgency']=='HIGH'])
        medium = len(history_df[history_df['Urgency']=='MEDIUM'])
        low    = len(history_df[history_df['Urgency']=='LOW'])

        s1,s2,s3,s4 = st.columns(4)
        with s1: st.markdown(f'<div class="stat-box" style="border-color:#2e86c1"><div style="font-size:0.8rem;color:#777">TOTAL</div><div style="font-size:2rem;font-weight:700;color:#2e86c1">{total}</div></div>', unsafe_allow_html=True)
        with s2: st.markdown(f'<div class="stat-box" style="border-color:#e74c3c"><div style="font-size:0.8rem;color:#777">🔴 HIGH</div><div style="font-size:2rem;font-weight:700;color:#e74c3c">{high}</div></div>', unsafe_allow_html=True)
        with s3: st.markdown(f'<div class="stat-box" style="border-color:#f39c12"><div style="font-size:0.8rem;color:#777">🟡 MEDIUM</div><div style="font-size:2rem;font-weight:700;color:#f39c12">{medium}</div></div>', unsafe_allow_html=True)
        with s4: st.markdown(f'<div class="stat-box" style="border-color:#27ae60"><div style="font-size:0.8rem;color:#777">🟢 LOW</div><div style="font-size:2rem;font-weight:700;color:#27ae60">{low}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        f1,f2 = st.columns([1,2])
        with f1: filter_urgency = st.selectbox("Filter by Urgency", ["All","HIGH","MEDIUM","LOW"])
        with f2: search_name = st.text_input("🔍 Search by name")

        filtered = history_df.copy()
        if filter_urgency != "All": filtered = filtered[filtered['Urgency']==filter_urgency]
        if search_name: filtered = filtered[filtered['Name'].str.contains(search_name, case=False, na=False)]

        def color_urgency(val):
            return {"HIGH":"background-color:#fdf2f2;color:#e74c3c;font-weight:bold",
                    "MEDIUM":"background-color:#fef9f0;color:#f39c12;font-weight:bold",
                    "LOW":"background-color:#f0faf4;color:#27ae60;font-weight:bold"}.get(val,"")

        st.markdown(f"**Showing {len(filtered)} record(s)**")
        st.dataframe(filtered.style.applymap(color_urgency, subset=['Urgency']), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.download_button("⬇️ Download History as CSV", filtered.to_csv(index=False).encode(), f"triage_history_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)

        log_audit(st.session_state.username, "HISTORY_VIEWED", f"Viewed {len(filtered)} records")

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — AUDIT LOG (admin + doctor only)
# ════════════════════════════════════════════════════════════════════════
with tab_audit:
    if st.session_state.role not in ["admin", "doctor"]:
        st.warning("🔒 Access restricted. Only Admin and Doctor roles can view the audit log.")
    else:
        st.subheader("📜 Security Audit Log")
        st.caption("Every login, assessment, and data access is recorded here.")
        audit_entries = load_audit_log()
        if not audit_entries:
            st.info("No audit entries yet.")
        else:
            audit_df = pd.DataFrame(audit_entries)
            # Filter
            action_filter = st.selectbox("Filter by Action", ["All"] + sorted(audit_df['Action'].unique().tolist()))
            if action_filter != "All":
                audit_df = audit_df[audit_df['Action'] == action_filter]

            def color_audit(val):
                if "SUCCESS" in str(val) or "ASSESSED" in str(val): return "color:#27ae60;font-weight:bold"
                if "FAILED" in str(val) or "BLOCKED" in str(val) or "LOCKED" in str(val) or "ATTEMPT" in str(val): return "color:#e74c3c;font-weight:bold"
                if "TIMEOUT" in str(val) or "LOGOUT" in str(val): return "color:#f39c12"
                return ""

            st.dataframe(audit_df.style.applymap(color_audit, subset=['Action']), use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Audit Log", audit_df.to_csv(index=False).encode(), "audit_log.csv", "text/csv", use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 4 — USER MANAGEMENT (admin only)
# ════════════════════════════════════════════════════════════════════════
if tab_admin and st.session_state.role == "admin":
    with tab_admin:
        st.subheader("👥 User Management")
        st.markdown("#### ➕ Create New User")

        with st.form("create_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role     = st.selectbox("Role", ["nurse","doctor","admin"])
            create_btn   = st.form_submit_button("Create User", type="primary")

        if create_btn:
            u_val = validate_username(new_username)
            p_val = validate_password_strength(new_password)

            if not u_val["valid"]:
                st.error(u_val["message"])
            elif not p_val["valid"]:
                st.error(p_val["message"])
            else:
                success = create_user(new_username, new_password, new_role)
                if success:
                    log_audit(st.session_state.username, "USER_CREATED", f"Created user: {new_username} role:{new_role}")
                    st.success(f"✅ User '{new_username}' created successfully!")
                else:
                    st.error(f"Username '{new_username}' already exists.")

        st.markdown("---")
        st.markdown("#### 🔑 Password Requirements")
        st.markdown("- Minimum 8 characters\n- At least 1 uppercase letter\n- At least 1 lowercase letter\n- At least 1 number\n- At least 1 special character (!@#$...)")
