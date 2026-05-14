# readmin-icu-discharge-risk-bot

# READMIN — ICU Readmission Risk & Intervention System

## ICU Discharge Risk & Intervention Bot

**MS Health Informatics — Hofstra University**  
MIMIC-IV Demo v2.2 · May 2026

READMIN is an end-to-end ICU readmission risk prediction and intervention platform built using machine learning, clinical workflow automation, and a Streamlit-based conversational interface.

The system predicts ICU readmission risk at the time of discharge and converts those predictions into structured patient follow-up workflows and clinician review tools.

---

# Project Overview

Hospital readmissions remain one of the most expensive and preventable problems in healthcare. ICU patients are particularly vulnerable after discharge, with approximately 15–20% experiencing unplanned readmission within 30 days.

READMIN addresses this problem by combining:

- Machine learning readmission prediction
- Risk scoring (0–100)
- Tiered clinical interventions
- Explainable AI (SHAP)
- Patient-facing follow-up workflows
- Clinician dashboards
- Automated escalation logic

---

# Core Features

## Machine Learning Scoring Engine

The scoring engine processes structured ICU discharge features and predicts the probability of readmission.

### Model Capabilities
- 30-day ICU readmission prediction
- 60-day and 90-day validation windows
- 39 engineered clinical features
- SHAP-based explainability
- LOW / MEDIUM / HIGH risk tier assignment

### Models Evaluated
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Final selected model:
- Logistic Regression

---

# Dataset

READMIN uses the **MIMIC-IV Demo Dataset**, a de-identified ICU clinical dataset maintained by MIT and Beth Israel Deaconess Medical Center.

### Data Sources Used
- admissions
- patients
- icustays
- diagnoses_icd
- labevents
- prescriptions
- chartevents

### Clinical Data Processed
- vital signs
- laboratory values
- ICD-10 diagnoses
- medications
- ICU stay metrics
- comorbidity indicators
- time-series trends

The project uses de-identified demo data only and does not process live patient-identifiable information.

---

# Bot Layer Overview

The READMIN Bot Layer converts machine learning outputs into structured patient and clinician workflows.

The interface is implemented in Streamlit and includes:

1. Progress Dashboard
2. Patient Bot (Vina)
3. Clinician Dashboard (Vedi)
4. Workflow View
5. Files & JSON Contract

---

# Patient Bot (Vina)

The patient-facing bot conducts structured post-discharge check-ins.

Unlike free-text chatbots, the system uses predefined button-based responses to ensure:

- consistent patient input
- reduced ambiguity
- clinically bounded interactions
- safe workflow control

## Patient Workflow

The patient check-in includes:

1. Medication adherence
2. Breathing score (1–5)
3. Symptom evaluation
4. Follow-up appointment verification

The conversation automatically branches based on patient responses.

---

# Escalation Logic

READMIN contains a structured escalation system.

## 🔴 Red Escalation
Triggered when:
- urgent symptoms reported
- breathing score ≤ 2
- severe medication concern

Actions:
- conversation stops immediately
- clinician notification triggered
- escalation message displayed

---

## 🟡 Yellow Warning
Triggered when:
- breathing below safe range
- mild symptoms present
- follow-up appointment missing or uncertain

Actions:
- patient urged to contact care team
- follow-up reminder displayed

---

## 🟢 Green Completion
Triggered when:
- responses remain within normal range
- follow-up appointment confirmed
- no escalation criteria met

Actions:
- successful check-in completion
- interaction logged

---

# Clinician Dashboard (Vedi)

The clinician-facing dashboard is designed for oversight and review.

## Dashboard Functions
- High-risk patient review
- SHAP-based risk explanations
- Workflow status monitoring
- Patient conversation review
- Chat log export

The dashboard surfaces the highest-risk ICU discharges first and provides clinicians with the context needed for intervention.

---

# Chat Logging & Audit Trail

Every patient interaction is stored in:

```text
logs/patient_chat_log.csv

```

---

# Project Structure
```text
  READMIN/
│
├── app/
│   ├── models/
│   │   ├── risk_model.pkl
│   │   ├── feature_cols.pkl
│   │   └── imputer.pkl
│   │
│   ├── csv/
│   │   ├── features_matrix.csv
│   │   └── patient_risk_scores.csv
│   │
│   ├── charts/
│   │   └── evaluation charts
│   │
│   ├── logs/
│   │   └── patient_chat_log.csv
│   │
│   └── bot-demo.py
│
├── requirements.txt
├── report/
├── Group_submission/
├── Data/
└── README.md

```

---

# Example JSON Request

```text
{
  "subject_id": 10031757,
  "stay_id": 123456,
  "features": {
    "anchor_age": 65,
    "gender_male": 1,
    "icu_los_days": 4.8,
    "n_prior_icu": 2,
    "comorbidity_score": 4
  }
}
```
---


# Example JSON Response

```text
{
  "risk_score": 82.4,
  "risk_tier": "HIGH",
  "readmit_prob_30d": 0.824,
  "top_drivers": [
    "Respiratory rate trend",
    "BUN / creatinine",
    "Prior ICU admissions"
  ],
  "interventions": [
    "Nurse call within 24 hours",
    "PCP appointment within 7 days",
    "Care coordination referral",
    "Daily SMS monitoring"
  ]
}
```

---
# Current Limitations
- Uses MIMIC-IV Demo dataset (100 patients)
- No live EHR integration
- No prospective clinical validation
- No FHIR connectivity yet
- Tiered interventions only (not individualized treatment recommendations)

---
# Authors

- Shreedhar Patel
- Aaliya Ailan
- Timothy Menz
- Imran Khan
- Dhruvkumar Rathod
- Keving Pierre
- Jinalkumari Vaniya
- Happy Patel
