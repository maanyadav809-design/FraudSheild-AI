# FRAUDSHIELD AI // Autonomous Cyber Defense Console

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![HTML5/CSS3/JS](https://img.shields.io/badge/Frontend-Vanilla_HTML_CSS_JS-00f0ff?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org)
[![Netlify](https://img.shields.io/badge/Deployed-Netlify-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://netlify.com)

**FraudShield AI** is an end-to-end, full-stack machine learning web application architected to detect and intercept fraudulent digital payment transactions in real time. Built with a high-contrast **Robotics + Futuristic Cyberpunk HUD** interface, it combines a responsive frontend console with a trained **Logistic Regression** classifier hosted on a lightweight **Flask** backend.

---

## Table of Contents
1. [End-to-End Webflow & System Architecture](#end-to-end-webflow--system-architecture)
2. [Project Directory Structure](#project-directory-structure)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
4. [Key Features](#key-features)
5. [Installation & Quickstart Guide](#installation--quickstart-guide)
6. [API Specification](#api-specification)
7. [Deployment (Netlify & Production)](#deployment-netlify--production)

---

## End-to-End Webflow & System Architecture

The following diagram illustrates how telemetry data flows from the user interface down to the machine learning inference engine and back:

```text
+-----------------------------------------------------------------------------------------+
|                                    CLIENT BROWSER                                       |
|                                                                                         |
|  [User / Analyst] ──> Enters Transaction Features or Clicks Attack Preset               |
|                           │                                                             |
|                           ▼                                                             |
|               [script.js - Client Controller]                                           |
|                           │                                                             |
|         Phase 1: Synthesizes Web Audio click / HUD sound                                |
|         Phase 2: Packs form inputs into JSON telemetry payload                          |
|         Phase 3: Initiates animated radar scanning HUD                                  |
|         Phase 4: Dispatches async POST request via fetch()                              |
+---------------------------│-------------------------------------------------------------+
                            │  HTTP POST /predict  (Payload: JSON)
                            ▼
+-----------------------------------------------------------------------------------------+
|                                  FLASK SERVER (app.py)                                  |
|                                                                                         |
|  [Route: /predict] ──> Validates JSON & checks required feature keys                    |
|                           │                                                             |
|                           ▼                                                             |
|               [Imputation & Normalization]                                              |
|         Fills any missing fields with training medians (e.g., amount: 1191.20)          |
|                           │                                                             |
|                           ▼                                                             |
|               [StandardScaler Transformation]                                           |
|         Transforms raw inputs: Z = (X - μ) / σ                                          |
|                           │                                                             |
|                           ▼                                                             |
|               [LogisticRegression Inference]                                            |
|         Computes Log-Odds: z = w^T · X + b                                              |
|         Calculates Sigmoid Probability: P(Fraud|X) = 1 / (1 + e^-z)                     |
|         Assigns Binary Class: 0 (Legitimate) or 1 (Fraud)                              |
|                           │                                                             |
|                           ▼                                                             |
|               [Explainable AI (XAI) Attribution Engine]                                 |
|         Calculates individual feature impact scores: w_i · x_scaled_i                   |
|         Generates mitigation policy recommendation (Quarantine / Challenge / Clear)    |
|                           │                                                             |
|                           ▼                                                             |
|               Returns HTTP 200 JSON Response payload                                    |
+---------------------------│-------------------------------------------------------------+
                            │  HTTP 200 OK  (JSON: probability, verdict, weights, etc.)
                            ▼
+-----------------------------------------------------------------------------------------+
|                                    CLIENT BROWSER                                       |
|                                                                                         |
|  [script.js - Response Handler]                                                         |
|         ├── Hides scanning radar HUD and reveals Results Panel                          |
|         ├── Animates Circular SVG Risk Gauge (0.0% -> P%) with color transition         |
|         ├── Renders Dynamic Verdict Banner (Safe: Green / Warn: Amber / Danger: Red)    |
|         ├── Updates Statistical Metrics ($Z$-score, Outlier status, Hour penalty)       |
|         ├── Renders Explainable AI (XAI) Feature Attribution weight bars                |
|         ├── Appends forensic event entry into Event Stream Terminal window              |
|         └── Plays adaptive audio tone (Safe chime or Critical alert horn)               |
+-----------------------------------------------------------------------------------------+
```

### Detailed Step-by-Step Breakdown:

1. **Ingestion & User Input:**  
   The user inputs transaction telemetry manually (Amount, Hour, Customer Age, Account Age, Previous Txns, Failed PIN Attempts, New Device toggle, Location Change toggle) or clicks one of the instant test vector presets:
   - **Normal Daytime (Safe):** ₹1,450 daytime purchase on a recognized device.
   - **Geo-Hop & New Device:** ₹12,800 night transaction with changed IP.
   - **₹52K Midnight Attack:** ₹52,000 at 03:00 AM, 5 failed OTP retries, new device, changed location, 45-day account (Student Project Challenge).

2. **Client Packaging & Asynchronous Request:**  
   When the analyst clicks **INITIATE AI NEURAL SCAN**:
   - `script.js` activates the Web Audio API synthesizer for a cybernetic sound pulse.
   - The scanning radar HUD displays animated telemetry progress steps.
   - The transaction vector is serialized into JSON and sent via `fetch('/predict', { method: 'POST', body: ... })`.

3. **Backend Parsing & Machine Learning Inference:**  
   - Flask's `/predict` route intercepts the request.
   - The feature vector is aligned with the schema from `models/fraud_model.joblib`.
   - The `StandardScaler` standardizes each feature against the population mean and variance.
   - The fitted `LogisticRegression` model computes:
     $$\text{Probability} = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
   - The model assigns a binary prediction (`1` for Fraud, `0` for Legitimate).
   - The backend computes Explainable AI feature weights:
     $$\text{Impact}_i = w_i \times x_{\text{scaled}, i}$$
     This highlights *why* the transaction was flagged (e.g., failed attempts contributed $+82.1\%$ risk).

4. **Dynamic Frontend Rendering & Enforcement:**  
   - The frontend transitions from scanning to results state.
   - The circular SVG radial gauge dynamically animates its stroke offset to the exact risk percentage.
   - Action buttons allow the analyst to execute immediate protocol responses:
     - **Quarantine Account:** Invalidate auth tokens and freeze settlement.
     - **Force Step-Up 2FA:** Dispatch biometric / SMS challenge.
     - **Whitelist TXN:** Manual clearance override.

5. **Graceful Offline Fallback:**  
   If the Python server is not reachable (such as when running as a static site on Netlify), `script.js` automatically catches the network exception and executes a client-side Bayesian & $Z$-score heuristic inference engine. The site never crashes or displays a blank screen.

---

## Project Directory Structure

```text
FraudShield AI/
├── app.py                                   # Flask backend & ML REST API
├── train_model.py                           # Training & model serialization pipeline
├── requirements.txt                         # Python package dependencies
├── notes.css                                # Data science concepts & training notes
├── README.md                                # Comprehensive project documentation
│
├── data/
│   └── AI_Fraud_Detection_Transactions.csv # Cleaned training dataset (8,012 rows)
│
├── models/
│   └── fraud_model.joblib                  # Serialized ML pipeline (Model, Scaler, Medians)
│
├── templates/
│   └── index.html                          # Flask HTML5 template with cyber aesthetics
│
└── static/
    ├── style.css                            # Robotic HUD theme, neon glows & responsive grid
    └── script.js                            # Interactive client engine & Web Audio synthesizer
```

---

## Machine Learning Pipeline

The core intelligence is driven by a supervised binary classification pipeline trained in [train_model.py](file:///c:/Users/rekha/OneDrive/Desktop/AIML%20project/FraudShield%20AI/train_model.py):

### 1. Data Cleaning & Preprocessing
- **Missing Value Imputation:** 
  - `amount`: Imputed with median ₹1,191.20
  - `customer_age`: Imputed with median 34 years
  - `previous_transactions`: Imputed with median 5
- **Feature Scaling:** `StandardScaler` standardizes numerical inputs to zero mean and unit variance.

### 2. Handling Class Imbalance
Fraud represents only **1.78%** of the dataset (143 fraud vs 7,869 legit). A naive model predicting every transaction as legitimate would achieve 98.22% accuracy while detecting **0** frauds.  
To resolve this, our `LogisticRegression` uses `class_weight='balanced'`, which automatically penalizes misclassifications of minority fraud instances inversely proportional to their class frequency.

### 3. Model Evaluation (20% Holdout Test Set)
- **ROC-AUC Score:** **0.7366**
- **Recall (Sensitivity):** **65.52%** (Catches nearly 2 out of 3 frauds on completely unseen holdout data)
- **Accuracy:** **72.68%**

### 4. Learned Feature Coefficients (Model Weights)
| Feature | Coefficient | Risk Direction |
|---|---|---|
| **`new_device`** | **+0.6345** | 🔴 Strongly Increases Fraud Risk |
| **`failed_attempts`** | **+0.4732** | 🔴 Strongly Increases Fraud Risk |
| **`location_changed`** | **+0.3597** | 🔴 Increases Fraud Risk |
| **`amount`** | **+0.0983** | 🔴 Increases Fraud Risk |
| **`customer_age`** | **-0.0590** | 🟢 Lowers Fraud Risk |
| **`previous_transactions`** | **-0.1580** | 🟢 Lowers Fraud Risk |
| **`account_age_days`** | **-0.2444** | 🟢 Lowers Fraud Risk |
| **`transaction_hour`** | **-0.5005** | 🟢 Daytime Hours Lower Risk |

---

## Key Features

1. **Robotics + Futuristic Design System:**
   - Obsidian dark glassmorphism backgrounds (`#03060f`, `#070d1c`)
   - Electric cyan (`#00f0ff`), neon emerald (`#10b981`), and plasma crimson (`#ff0055`) accents.
   - Google Fonts: [Orbitron](https://fonts.google.com/specimen/Orbitron), [Rajdhani](https://fonts.google.com/specimen/Rajdhani), and [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono).
   - Interactive HTML5 Canvas neural particle mesh background.
2. **Zero-Dependency Web Audio Synthesizer:**
   - Real-time synthetic sound effects generated dynamically using the browser's `AudioContext` (clicks, scanning sweeps, alert sirens) with an audio mute toggle in the navbar.
3. **Telemetry HUD Summary Cards:**
   - Real-time global transaction counter, intercepted threats counter, model precision gauge, and inference latency ticker.
4. **Live Event Stream Terminal:**
   - Scrolling monospace terminal displaying forensic analysis logs, system timestamps, and JSON export capabilities.

---

## Installation & Quickstart Guide

### Prerequisites
- Python 3.10+ installed
- Web browser (Chrome, Edge, Firefox, Brave)

### 1. Clone the Repository
```bash
git clone https://github.com/maanyadav809-design/FraudSheild-AI.git
cd FraudSheild-AI/"FraudShield AI"
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Machine Learning Model
```bash
python train_model.py
```
*This processes `data/AI_Fraud_Detection_Transactions.csv` and serializes the updated pipeline to `models/fraud_model.joblib`.*

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://localhost:5000
```

---

## API Specification

### 1. Health & Model Status
`GET /api/health`

**Response (200 OK):**
```json
{
  "status": "online",
  "service": "FraudShield AI Sentinel Core",
  "model_loaded": true,
  "model_type": "Logistic Regression Classifier",
  "metrics": {
    "accuracy": 0.7268,
    "f1_score": 0.0798,
    "precision": 0.0425,
    "recall": 0.6552,
    "roc_auc": 0.7366
  }
}
```

### 2. Transaction Inference
`POST /predict` (or `/api/analyze`)

**Request Payload:**
```json
{
  "transaction_id": "TXN-FINAL-52K",
  "amount": 52000.0,
  "transaction_hour": 3,
  "customer_age": 26,
  "previous_transactions": 0,
  "failed_attempts": 5,
  "new_device": 1,
  "location_changed": 1,
  "account_age_days": 45
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "model": "Logistic Regression Classifier (Scikit-Learn)",
  "transaction_id": "TXN-FINAL-52K",
  "prediction": 1,
  "is_fraud": true,
  "fraud_probability": 0.9998,
  "risk_score_percent": 99.98,
  "risk_level": "CRITICAL FRAUD",
  "verdict_badge": "CRITICAL THREAT",
  "verdict_title": "ALERT // FRAUD THREAT DETECTED",
  "verdict_description": "High probability of fraudulent behavior identified by Logistic Regression classifier.",
  "recommendation": "AUTOMATED QUARANTINE: Freeze transaction settlement and notify account holder immediately.",
  "feature_attributions": [
    { "feature": "failed_attempts", "impact": 3.731, "direction": "danger", "weight_pct": 82.1 },
    { "feature": "amount", "impact": 1.796, "direction": "danger", "weight_pct": 39.5 },
    { "feature": "new_device", "impact": 1.377, "direction": "danger", "weight_pct": 30.3 },
    { "feature": "location_changed", "impact": 0.994, "direction": "danger", "weight_pct": 21.9 },
    { "feature": "transaction_hour", "impact": 0.626, "direction": "danger", "weight_pct": 13.8 }
  ]
}
```

---

## Deployment (Netlify & Production)

### Netlify Deployment Configuration
The repository includes a root `netlify.toml` file configured for static frontend deployment:
```toml
[build]
  publish = "FraudShield AI"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```
When hosted statically on Netlify, the frontend automatically detects the absence of the Flask backend and switches to its high-precision client-side Bayesian heuristic engine.

---

## License
MIT License. Created for digital payments fraud investigation and AI machine learning engineering.
