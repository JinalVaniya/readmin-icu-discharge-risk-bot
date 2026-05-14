from __future__ import annotations

from datetime import datetime
from pathlib import Path
import uuid

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='IDRIB Bot Mockup', page_icon='🩺', layout='wide')

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
CSV_DIR = BASE_DIR / 'csv'
CHARTS_DIR = BASE_DIR / 'charts'
LOGS_DIR = BASE_DIR / 'logs'

MODEL_PATH = MODELS_DIR / 'risk_model.pkl'
FEATURE_COLS_PATH = MODELS_DIR / 'feature_cols.pkl'
IMPUTER_PATH = MODELS_DIR / 'imputer.pkl'
FEATURES_MATRIX_PATH = CSV_DIR / 'features_matrix.csv'
RISK_OUTPUT_PATH = CSV_DIR / 'patient_risk_scores.csv'
CHAT_LOG_PATH = LOGS_DIR / 'patient_chat_log.csv'

for folder in [MODELS_DIR, CSV_DIR, CHARTS_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #ffffff 140%);
        color: white;
        padding: 1.5rem;
        border-radius: 24px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
    }
    .card {
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1rem;
        background: white;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .bubble-bot {
        background: #e2e8f0;
        color: #0f172a;
        padding: 0.9rem 1rem;
        border-radius: 18px 18px 18px 6px;
        margin-bottom: 0.7rem;
        max-width: 90%;
        font-size: 18px;
        line-height: 1.5;
    }
    .bubble-user {
        background: #0f172a;
        color: white;
        padding: 0.9rem 1rem;
        border-radius: 18px 18px 6px 18px;
        margin-left: auto;
        margin-bottom: 0.7rem;
        max-width: 90%;
        font-size: 18px;
        line-height: 1.5;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .high { background: #ffe4e6; color: #be123c; }
    .med { background: #fef3c7; color: #92400e; }
    .low { background: #dcfce7; color: #166534; }
    .muted { color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div style="font-size:0.82rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#cbd5e1;">IDRIB — ICU Discharge Risk & Intervention Bot</div>
        <h1 style="margin:0.35rem 0 0.5rem 0; font-size:2.1rem;">Patient-clinician front-end prototype</h1>
        <p style="margin:0; color:#cbd5e1;">Patient bot, clinician dashboard, workflow view, and JSON contract in one Python app.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    feature_cols = joblib.load(FEATURE_COLS_PATH) if FEATURE_COLS_PATH.exists() else []
    imputer = joblib.load(IMPUTER_PATH) if IMPUTER_PATH.exists() else None
    return model, feature_cols, imputer


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


model, feature_cols, imputer = load_model_assets()
features_df = load_csv(FEATURES_MATRIX_PATH)
risk_df = load_csv(RISK_OUTPUT_PATH)
chat_df = load_csv(CHAT_LOG_PATH)

view = st.sidebar.selectbox(
    'View',
    ['Progress Dashboard', 'Patient Bot', 'Clinician Dashboard', 'Workflow View', 'Files & JSON Contract']
)

st.sidebar.markdown('### Demo controls')
score = st.sidebar.slider('Risk score', 0, 100, 82)
subject_id = st.sidebar.text_input('Subject ID', '10031757')
stay_id = st.sidebar.text_input('Stay ID', '123456')
outtime = st.sidebar.text_input('Discharge time', '2137-10-14T17:08:00')

progress_items = {
    'Data loading': 100,
    'Feature engineering': 100,
    'Model training': 100,
    'Calibration': 75,
    'API wrapper': 65 if not features_df.empty else 55,
    'Patient bot': 60 if not features_df.empty else 45,
    'Clinician dashboard': 70 if not features_df.empty else 60,
    'Workflow automation': 50 if not features_df.empty else 40,
}

st.sidebar.markdown('### Project progress')
for label, value in progress_items.items():
    st.sidebar.write(label)
    st.sidebar.progress(value / 100.0)

if score >= 70:
    tier = 'HIGH'
    tier_class = 'high'
elif score >= 40:
    tier = 'MEDIUM'
    tier_class = 'med'
else:
    tier = 'LOW'
    tier_class = 'low'

interventions = {
    'HIGH': [
        'Nurse call within 24 hours',
        'PCP appointment within 7 days',
        'Care coordination referral',
        'Daily SMS monitoring',
    ],
    'MEDIUM': [
        '48-hour follow-up call',
        'PCP appointment within 14 days',
        'Pharmacist medication reconciliation',
    ],
    'LOW': [
        'Standard discharge instructions',
        'Patient portal message at day 7',
    ],
}

progress_df = pd.DataFrame(
    [
        {'Phase': 'Data loading', 'Completion': 100},
        {'Phase': 'Feature engineering', 'Completion': 100},
        {'Phase': 'Model training', 'Completion': 100},
        {'Phase': 'Calibration', 'Completion': 75},
        {'Phase': 'API wrapper', 'Completion': 65},
        {'Phase': 'Patient bot', 'Completion': 60},
        {'Phase': 'Clinician dashboard', 'Completion': 70},
        {'Phase': 'Workflow automation', 'Completion': 50},
    ]
)

trend_df = pd.DataFrame(
    {
        'Week': ['W1', 'W2', 'W3', 'W4', 'W5'],
        'Progress': [20, 38, 55, 72, 86],
    }
)

window_df = pd.DataFrame(
    {
        'Window': ['30-day', '60-day', '90-day'],
        'ROC_AUC': [0.607, 0.498, 0.696],
        'PR_AUC': [0.370, 0.307, 0.445],
    }
)

patient_options = {
    0: ['Yes', 'Partially', 'No'],
    1: ['5', '4', '3', '2', '1'],
    2: ['No symptoms', 'Mild symptoms', 'Urgent symptoms'],
    3: ['Yes', 'No', 'Not sure'],
}


def reset_patient_flow() -> None:
    st.session_state.patient_messages = [
        ('bot', 'Hi, I am checking in after your ICU discharge. Are you taking your medications as prescribed today?')
    ]
    st.session_state.patient_step = 0
    st.session_state.patient_done = False
    st.session_state.patient_escalated = False
    st.session_state.patient_breathing_score = None
    st.session_state.patient_symptoms = None
    st.session_state.patient_appointment_status = None
    st.session_state.session_id = uuid.uuid4().hex[:12]


def save_chat_event(role: str, message: str) -> None:
    row = pd.DataFrame(
        [
            {
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'session_id': st.session_state.get('session_id', 'unknown'),
                'subject_id': subject_id,
                'stay_id': stay_id,
                'step': st.session_state.get('patient_step', 0),
                'role': role,
                'message': message,
                'patient_done': st.session_state.get('patient_done', False),
                'patient_escalated': st.session_state.get('patient_escalated', False),
            }
        ]
    )
    row.to_csv(CHAT_LOG_PATH, mode='a', header=not CHAT_LOG_PATH.exists(), index=False)



def next_question(step: int, choice: str) -> str:
    choice = choice.lower()
    if step == 0:
        if choice == 'yes':
            return 'Great. On a scale of 1 to 5, how is your breathing today?'
        if choice == 'partially':
            return 'Thanks for telling me. Are you having any symptoms today? Rate on scale of 1 to 5.'
        if choice == 'no':
            return 'Urgent escalation: I am notifying the care team now.'
        return 'Please choose one of the medication options so I can continue.'
    if step == 1:
        if choice in ['5', '4']:
            return 'Good to hear. Do you have your follow-up appointment scheduled?'
        if choice == '3':
            return 'Thanks. Are you feeling worse than when you left the hospital?'
        if choice in ['2', '1']:
            return 'Urgent escalation: your breathing score is low. I am notifying the care team now.'
        return 'Please choose a breathing rating from 1 to 5.'
    if step == 2:
        if choice == 'no symptoms':
            return 'Great. Are you taking your medications as directed?'
        if choice == 'mild symptoms':
            return 'Thanks. Please keep monitoring your symptoms and confirm your follow-up appointment.'
        if choice == 'urgent symptoms':
            return 'Urgent escalation: I am notifying the care team now.'
        return 'Please choose a symptom option.'
    if step == 3:
        return 'Thank you. Your check-in is complete.'
    return 'Thank you. Your check-in is complete.'



def patient_followup_warning() -> str | None:
    breathing = st.session_state.get('patient_breathing_score')
    symptoms = st.session_state.get('patient_symptoms')
    appointment = st.session_state.get('patient_appointment_status')

    if breathing is None:
        return None

    if breathing < 3:
        if appointment in ['no', 'not sure']:
            return 'Please schedule a follow-up appointment as soon as possible and contact your care team today.'
        if symptoms in ['mild symptoms', 'urgent symptoms']:
            return 'Please contact your care team today and schedule your follow-up appointment as soon as possible.'

    if symptoms in ['mild symptoms', 'urgent symptoms'] and appointment in ['no', 'not sure']:
        return 'Please schedule a follow-up appointment as soon as possible and contact your care team today.'

    return None


if 'patient_messages' not in st.session_state:
    reset_patient_flow()

if view == 'Progress Dashboard':
    a, b, c, d = st.columns(4)
    a.metric('Core pipeline', '100%', 'Data, labels, features')
    b.metric('Modeling', '75%', 'Calibration and validation')
    c.metric('Bot layer', '65%', 'API + front end')
    d.metric('Deployment', '50%', 'Workflow automation')

    left, right = st.columns(2, gap='large')
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Build completion by phase')
        st.bar_chart(progress_df.set_index('Phase')['Completion'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Project roadmap trend')
        st.line_chart(trend_df.set_index('Week'))
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Risk tier distribution from patient_risk_scores.csv')
        if not risk_df.empty and 'risk_tier' in risk_df.columns:
            counts = risk_df['risk_tier'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH']).fillna(0)
            st.bar_chart(counts)
        else:
            st.info('patient_risk_scores.csv not found in csv/ folder.')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Validation windows')
        st.dataframe(window_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### Charts from the charts folder')
    chart_files = sorted(list(CHARTS_DIR.glob('*.png')) + list(CHARTS_DIR.glob('*.jpg')) + list(CHARTS_DIR.glob('*.jpeg')))
    if chart_files:
        for i in range(0, len(chart_files), 2):
            cols = st.columns(2)
            cols[0].image(str(chart_files[i]), use_container_width=True)
            if i + 1 < len(chart_files):
                cols[1].image(str(chart_files[i + 1]), use_container_width=True)
    else:
        st.info('No chart files found in charts/ folder.')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### Delivery status')
    for label, value in progress_items.items():
        st.write(f'**{label}**')
        st.progress(value / 100.0)
    st.markdown('</div>', unsafe_allow_html=True)

elif view == 'Patient Bot':
    left, right = st.columns([1, 1.2], gap='large')
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Readmission Risk Score')
        st.markdown(f'<div style="font-size:2rem; font-weight:800;">{score}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge {tier_class}">{tier} RISK</span>', unsafe_allow_html=True)
        st.progress(score / 100.0)
        st.caption('0 = low risk, 100 = highest risk')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Recommended interventions')
        for item in interventions[tier]:
            st.write(f'- {item}')
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Patient check-in conversation')
        st.caption('Use the buttons below. No typing is required.')

        for role, text in st.session_state.patient_messages:
            if role == 'bot':
                if 'Urgent escalation' in text:
                    st.markdown(
                        f'<div style="background:#fee2e2;color:#991b1b;padding:0.9rem 1rem;border-radius:18px 18px 18px 6px;margin-bottom:0.7rem;"><strong>Bot</strong><br>{text}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f'<div class="bubble-bot"><strong>Bot</strong><br>{text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bubble-user"><strong>Patient</strong><br>{text}</div>', unsafe_allow_html=True)

        if not st.session_state.patient_done:
            if st.session_state.patient_escalated:
                st.markdown(
                    '<div style="padding:0.9rem 1rem; border-radius:18px; border:1px solid #fecaca; background:#fef2f2; color:#991b1b;"><strong>Escalation status</strong><br>The care team has been notified. No further questions will be asked.</div>',
                    unsafe_allow_html=True,
                )
            else:
                current_step = st.session_state.patient_step
                labels = patient_options.get(current_step, [])
                if labels:
                    cols = st.columns(len(labels))
                    for i, label in enumerate(labels):
                        if cols[i].button(label, key=f'step_{current_step}_{label}'):
                            st.session_state.patient_messages.append(('user', label))
                            save_chat_event('patient', label)

                            bot_text = next_question(current_step, label)
                            st.session_state.patient_messages.append(('bot', bot_text))
                            save_chat_event('bot', bot_text)

                            if current_step == 0 and label == 'No':
                                st.session_state.patient_escalated = True
                                st.session_state.patient_done = True
                            elif current_step == 1:
                                st.session_state.patient_breathing_score = int(label)
                                if label in ['2', '1']:
                                    st.session_state.patient_escalated = True
                                    st.session_state.patient_done = True
                                else:
                                    st.session_state.patient_step += 1
                            elif current_step == 2:
                                st.session_state.patient_symptoms = label.lower()
                                if label == 'Urgent symptoms':
                                    st.session_state.patient_escalated = True
                                    st.session_state.patient_done = True
                                else:
                                    st.session_state.patient_step += 1
                            elif current_step == 3:
                                st.session_state.patient_appointment_status = label.lower()
                                st.session_state.patient_done = True
                            else:
                                st.session_state.patient_step += 1

                            warning = patient_followup_warning()
                            if warning:
                                st.session_state.patient_step = 4

                            st.rerun()

        warning = patient_followup_warning()
        if warning and not st.session_state.patient_escalated:
            st.markdown(
                '<div style="padding:0.9rem 1rem; border-radius:18px; border:1px solid #facc15; background:#fefce8; color:#854d0e;"><strong>Follow-up reminder</strong><br>' + warning + '</div>',
                unsafe_allow_html=True,
            )

        if st.session_state.patient_escalated:
            st.markdown(
                '<div style="padding:0.9rem 1rem; border-radius:18px; border:1px solid #fecaca; background:#fef2f2; color:#991b1b;"><strong>Urgent escalation</strong><br>The care team has been notified immediately.</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.patient_done:
            st.markdown(
                '<div style="padding:0.9rem 1rem; border-radius:18px; border:1px solid #bbf7d0; background:#f0fdf4; color:#166534;"><strong>Response status</strong><br>Completed check-in · no escalation needed right now</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Bot logic at discharge')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('**Input**')
            st.write('39 structured EHR features')
        with col2:
            st.write('**Processing**')
            st.write('Scoring API + SHAP explanation')
        with col3:
            st.write('**Output**')
            st.write('Patient task + clinician alert')
        st.markdown('</div>', unsafe_allow_html=True)

elif view == 'Clinician Dashboard':
    left, right = st.columns([1, 1.3], gap='large')
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Top discharge alerts')
        if not risk_df.empty and 'risk_score' in risk_df.columns:
            alert_rows = risk_df.sort_values('risk_score', ascending=False).head(3)
            for _, row in alert_rows.iterrows():
                t = str(row.get('risk_tier', 'LOW'))
                badge_class = 'high' if t == 'HIGH' else 'med' if t == 'MEDIUM' else 'low'
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0; border-radius:16px; padding:0.9rem; margin-bottom:0.7rem;">'
                    f'<div style="display:flex; justify-content:space-between; gap:1rem;"><strong>Subject {row.get("subject_id", "")}</strong><span class="badge {badge_class}">{t}</span></div>'
                    f'<div class="muted" style="margin-top:0.25rem;">Score {row.get("risk_score", "")} / 100</div>'
                    f'<div style="margin-top:0.45rem;">{row.get("top_risk_drivers", "")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            alerts = [
                ('Subject 10031757', 'HIGH', '97.1', 'Rising respiratory rate · prior ICU admission'),
                ('Subject 10024051', 'HIGH', '89.4', 'Creatinine trend · vasopressor flag'),
                ('Subject 10038878', 'MEDIUM', '58.2', 'Medication count · discharge destination'),
            ]
            for name, t, s, note in alerts:
                badge_class = 'high' if t == 'HIGH' else 'med'
                st.markdown(
                    f'<div style="border:1px solid #e2e8f0; border-radius:16px; padding:0.9rem; margin-bottom:0.7rem;">'
                    f'<div style="display:flex; justify-content:space-between; gap:1rem;"><strong>{name}</strong><span class="badge {badge_class}">{t}</span></div>'
                    f'<div class="muted" style="margin-top:0.25rem;">Score {s} / 100</div>'
                    f'<div style="margin-top:0.45rem;">{note}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Workflow status')
        st.write('1. Score generated')
        st.write('2. SHAP explanation attached')
        st.write('3. Intervention task queued')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Chat log for clinician review')
        if not chat_df.empty:
            recent = chat_df.sort_values('timestamp', ascending=False).head(25)
            st.dataframe(recent, use_container_width=True, hide_index=True)
            st.download_button(
                'Download chat log CSV',
                data=chat_df.to_csv(index=False).encode('utf-8'),
                file_name='patient_chat_log.csv',
                mime='text/csv',
            )
        else:
            st.info('No chat log found yet. Once the patient bot is used, messages will appear here.')
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Why this patient is high risk')
        drivers = [
            ('Respiratory rate trend', 92),
            ('BUN / creatinine', 81),
            ('Prior ICU admissions', 73),
            ('Discharge destination', 66),
        ]
        for label, value in drivers:
            st.write(f'**{label}** — {value}% impact')
            st.progress(value / 100.0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Dashboard modules')
        st.write('- Today’s discharges')
        st.write('- High-risk list')
        st.write('- Open outreach tasks')
        st.write('- Calibration / QC')
        st.markdown('</div>', unsafe_allow_html=True)

elif view == 'Workflow View':
    left, right = st.columns([1, 1.2], gap='large')
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Workflow overview')
        st.write('Step 1 — Discharge event')
        st.write('Step 2 — Model scoring')
        st.write('Step 3 — Task generation')
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Bot deployment flow')
        st.write('Input: 39 structured EHR features')
        st.write('Processing: Scoring API + SHAP explanation')
        st.write('Output: Patient task + clinician alert')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Recommended files')
        st.write('- risk_model.pkl')
        st.write('- feature_cols.pkl')
        st.write('- imputer.pkl')
        st.write('- features_matrix.csv')
        st.write('- patient_risk_scores.csv')
        st.write('- charts/')
        st.write('- bot-demo.py')
        st.markdown('</div>', unsafe_allow_html=True)

else:
    left, right = st.columns([1, 1], gap='large')
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Files the bot layer uses')
        st.write('- `models/risk_model.pkl`')
        st.write('- `models/feature_cols.pkl`')
        st.write('- `models/imputer.pkl`')
        st.write('- `csv/features_matrix.csv`')
        st.write('- `csv/patient_risk_scores.csv`')
        st.write('- `charts/` folder')
        st.write('- `bot-demo.py`')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Example JSON request')
        st.code(
            f'''{{
  "subject_id": {subject_id},
  "stay_id": {stay_id},
  "outtime": "{outtime}",
  "features": {{
    "anchor_age": 65,
    "gender_male": 1,
    "insurance_medicare": 1,
    "discharge_home": 0,
    "icu_los_days": 4.8,
    "n_prior_icu": 2,
    "comorbidity_score": 4,
    "elixhauser_score": 3,
    "ventilation_flag": 1,
    "vasopressor_flag": 1
  }}
}}''',
            language='json',
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### Example JSON response')
        st.code(
            '''{
  "risk_score": 82.4,
  "risk_tier": "HIGH",
  "readmit_prob_30d": 0.824,
  "top_drivers": [
    "respiratory rate trend",
    "BUN",
    "prior ICU admissions"
  ],
  "interventions": [
    "Nurse call within 24 hours",
    "PCP appointment within 7 days",
    "Care coordination referral",
    "Daily SMS monitoring"
  ]
}''',
            language='json',
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### How to run this app')
        st.code('streamlit run bot-demo.py', language='bash')
        st.markdown('</div>', unsafe_allow_html=True)
