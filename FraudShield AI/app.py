"""
=============================================================================
FRAUDSHIELD AI // FLASK BACKEND & MACHINE LEARNING INFERENCE SERVICE
=============================================================================
Built according to AIML Project curriculum & notes.css specifications:

Features:
  - Feature 1: AI Transaction Analyser (POST /api/analyze, POST /predict)
  - Feature 2: Interactive Analytics Dashboard (GET /api/analytics, GET /api/chart)
  - Feature 3: Suspicious Transaction Explorer (GET /api/transactions)

Tech Stack:
  - Frontend: HTML5, CSS3, JavaScript
  - Backend: Python, Flask
  - Data Science & ML: Pandas, NumPy, Scikit-learn (Logistic Regression), Matplotlib

API Request Types (notes.css):
  1. GET  Request -> Backend to Frontend (Telemetry, Analytics, Transaction Explorer)
  2. POST Request -> Frontend to Backend (Transaction Input -> ML Inference Response)
=============================================================================
"""

import os
import sys
import io
import base64
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# Set Matplotlib to non-interactive Agg backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows terminals
if sys.platform == 'win32' and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Directories & Path Configurations
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'AI_Fraud_Detection_Transactions.csv')

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR,
    static_url_path='/static'
)

# ---------------------------------------------------------------------------
# Global In-Memory Cache (ML Model & Dataset)
# ---------------------------------------------------------------------------
ml_pipeline = None
cached_df = None

def load_fraud_model():
    """Loads the serialized Logistic Regression pipeline from disk."""
    global ml_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            ml_pipeline = joblib.load(MODEL_PATH)
            print(f"[*] [MODEL LOADED] Logistic Regression model ready from: {MODEL_PATH}")
        except Exception as e:
            print(f"[!] [ERROR] Failed to load model artifact: {e}")
            ml_pipeline = None
    else:
        print(f"[!] [WARNING] Model artifact missing at {MODEL_PATH}. Run 'python train_model.py' first.")

def load_dataset():
    """Loads and caches the cleaned dataset for analytics and explorer features."""
    global cached_df
    if os.path.exists(DATA_PATH):
        try:
            cached_df = pd.read_csv(DATA_PATH)
            print(f"[*] [DATASET LOADED] Ingested {len(cached_df)} records from {DATA_PATH}")
        except Exception as e:
            print(f"[!] [ERROR] Failed to read dataset: {e}")
            cached_df = None

# Initialize on startup
load_fraud_model()
load_dataset()

# ===========================================================================
# ROUTE: MAIN DASHBOARD VIEW
# ===========================================================================
@app.route('/')
def home():
    """Renders the primary Cyber-Robotic UI Console."""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """System health check and active ML model telemetry."""
    is_ready = ml_pipeline is not None
    return jsonify({
        "status": "online" if is_ready else "degraded",
        "service": "FraudShield AI Sentinel Core",
        "model_loaded": is_ready,
        "model_type": "Logistic Regression (Binary Classifier)",
        "features_tracked": ml_pipeline.get('feature_names') if is_ready else [],
        "metrics": ml_pipeline.get('evaluation_metrics') if is_ready else None
    })

# ===========================================================================
# FEATURE ONE -> AI TRANSACTION ANALYSER
# Workflow (notes.css):
# User enters transaction -> clicks analyse button -> HTTP POST request (Flask API)
# -> receives input data -> processes features -> Logistic Regression predicts risk
# -> JSON response (prediction, probability, XAI weights) -> results on web
# ===========================================================================
@app.route('/predict', methods=['POST'])
@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    """
    Feature 1: AI Transaction Analyser
    Accepts transaction features via POST request and returns ML prediction.
    """
    global ml_pipeline
    if ml_pipeline is None:
        load_fraud_model()
        if ml_pipeline is None:
            return jsonify({
                "status": "error",
                "message": "Logistic Regression model not loaded. Please run 'python train_model.py' first."
            }), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON transaction payload"}), 400

        # Step 1: Extract features with fallbacks to training medians
        impute_vals = ml_pipeline.get('impute_values', {})
        amount = float(data.get('amount', impute_vals.get('amount', 1191.19)))
        hour = int(data.get('transaction_hour', data.get('hour', 12)))
        age = int(data.get('customer_age', impute_vals.get('customer_age', 34)))
        prev_txns = int(data.get('previous_transactions', impute_vals.get('previous_transactions', 5)))
        failed = int(data.get('failed_attempts', 0))

        # Binary categorical indicators (0 or 1)
        new_dev = 1 if data.get('new_device') in [1, True, '1', 'true', 'True'] else 0
        loc_changed = 1 if data.get('location_changed') in [1, True, '1', 'true', 'True'] else 0
        acc_age = int(data.get('account_age_days', data.get('account_age', 180)))
        tx_id = str(data.get('transaction_id', data.get('txId', 'TXN-00000')))

        # Step 2: Format into feature DataFrame matching model schema
        feature_cols = ml_pipeline['feature_names']
        feature_dict = {
            'amount': amount,
            'transaction_hour': hour,
            'customer_age': age,
            'previous_transactions': prev_txns,
            'failed_attempts': failed,
            'new_device': new_dev,
            'location_changed': loc_changed,
            'account_age_days': acc_age
        }
        X_df = pd.DataFrame([feature_dict])[feature_cols]

        # Step 3: Feature Standardization (StandardScaler)
        scaler = ml_pipeline['scaler']
        model = ml_pipeline['model']
        X_scaled = scaler.transform(X_df)

        # Step 4: Logistic Regression Inference
        # Computes Sigmoid probability: P(Fraud) = 1 / (1 + e^-z)
        pred_class = int(model.predict(X_scaled)[0])
        pred_prob = float(model.predict_proba(X_scaled)[0, 1])
        risk_score_pct = round(pred_prob * 100, 2)

        # Step 5: Risk Stratification & Protocol Recommendation
        if risk_score_pct >= 65.0 or pred_class == 1:
            risk_level = "CRITICAL FRAUD"
            verdict_badge = "CRITICAL THREAT"
            verdict_title = "ALERT // FRAUD THREAT DETECTED"
            verdict_desc = "High probability of fraudulent behavior identified by Logistic Regression classifier."
            rec_action = "AUTOMATED QUARANTINE: Freeze transaction settlement and notify account holder immediately."
        elif risk_score_pct >= 30.0:
            risk_level = "ELEVATED RISK"
            verdict_badge = "SUSPICIOUS"
            verdict_title = "FLAGGED // ELEVATED RISK TRANSACTION"
            verdict_desc = "Moderate statistical deviation from typical customer baseline. Step-up MFA verification advised."
            rec_action = "CHALLENGE REQUIRED: Dispatch 3D Secure / Biometric OTP challenge before releasing payment."
        else:
            risk_level = "LOW RISK"
            verdict_badge = "CLEARED"
            verdict_title = "CLEARED // TRANSACTION AUTHORIZED"
            verdict_desc = "Nominal transaction signature. Feature weights align with authorized customer behavioral patterns."
            rec_action = "SETTLEMENT APPROVED: Proceed with straight-through gateway execution."

        # Step 6: Explainable AI (XAI) Attribution Breakdown
        coefficients = ml_pipeline['coefficients']
        scaled_vals = X_scaled[0]
        attributions = []

        for feat_name, s_val in zip(feature_cols, scaled_vals):
            coef = coefficients.get(feat_name, 0.0)
            impact = coef * s_val
            if abs(impact) > 0.05:
                direction = "danger" if impact > 0 else "safe"
                attributions.append({
                    "feature": feat_name,
                    "impact": round(impact, 3),
                    "direction": direction,
                    "weight_pct": min(round(abs(impact) * 22, 1), 95)
                })

        attributions.sort(key=lambda x: abs(x['impact']), reverse=True)

        return jsonify({
            "status": "success",
            "feature": "Feature 1: AI Transaction Analyser",
            "model": "Logistic Regression Classifier (Scikit-Learn)",
            "transaction_id": tx_id,
            "prediction": pred_class,
            "is_fraud": bool(pred_class == 1),
            "fraud_probability": round(pred_prob, 4),
            "risk_score_percent": risk_score_pct,
            "risk_level": risk_level,
            "verdict_badge": verdict_badge,
            "verdict_title": verdict_title,
            "verdict_description": verdict_desc,
            "recommendation": rec_action,
            "feature_attributions": attributions,
            "inputs_evaluated": feature_dict
        })

    except Exception as e:
        print(f"[!] Error in analyze_transaction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ===========================================================================
# FEATURE TWO -> INTERACTIVE ANALYTICS DASHBOARD
# Computes distribution, variability, outliers, correlation, and summary stats
# ===========================================================================
@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """
    Feature 2: Interactive Analytics Dashboard
    Calculates dataset distributions, statistical variability, and fraud metrics using Pandas & NumPy.
    """
    global cached_df
    if cached_df is None:
        load_dataset()
        if cached_df is None:
            return jsonify({"status": "error", "message": "Dataset unavailable"}), 503

    df = cached_df.copy()
    amount_clean = df['amount'].dropna()

    # Core statistics (mean, median, std, variability)
    total_txns = len(df)
    fraud_counts = df['fraud'].value_counts()
    fraud_count = int(fraud_counts.get(1, 0))
    legit_count = int(fraud_counts.get(0, 0))
    fraud_rate = round((fraud_count / total_txns) * 100, 2)

    mean_amt = float(amount_clean.mean())
    median_amt = float(amount_clean.median())
    std_amt = float(amount_clean.std())
    q1 = float(amount_clean.quantile(0.25))
    q3 = float(amount_clean.quantile(0.75))
    iqr = q3 - q1

    # Breakdown by fraud status (legit vs fraud)
    grouped = df.groupby('fraud')['amount'].agg(['mean', 'median', 'std', 'count']).to_dict(orient='index')

    legit_stats = {
        "count": int(grouped.get(0, {}).get('count', 0)),
        "mean_amount": round(float(grouped.get(0, {}).get('mean', 0)), 2),
        "median_amount": round(float(grouped.get(0, {}).get('median', 0)), 2),
        "std_amount": round(float(grouped.get(0, {}).get('std', 0)), 2)
    }

    fraud_stats = {
        "count": int(grouped.get(1, {}).get('count', 0)),
        "mean_amount": round(float(grouped.get(1, {}).get('mean', 0)), 2),
        "median_amount": round(float(grouped.get(1, {}).get('median', 0)), 2),
        "std_amount": round(float(grouped.get(1, {}).get('std', 0)), 2)
    }

    # Hourly fraud distribution
    hourly_fraud = df.groupby('transaction_hour')['fraud'].agg(['count', 'sum'])
    hourly_fraud['fraud_rate'] = (hourly_fraud['sum'] / hourly_fraud['count']) * 100
    top_risk_hours = hourly_fraud.sort_values(by='fraud_rate', ascending=False).head(4).index.tolist()

    return jsonify({
        "status": "success",
        "feature": "Feature 2: Interactive Analytics Dashboard",
        "dataset_summary": {
            "total_transactions": total_txns,
            "legitimate_count": legit_count,
            "fraudulent_count": fraud_count,
            "fraud_rate_percent": fraud_rate
        },
        "amount_statistics": {
            "mean": round(mean_amt, 2),
            "median": round(median_amt, 2),
            "std": round(std_amt, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "outlier_threshold": round(q3 + 1.5 * iqr, 2)
        },
        "segment_comparison": {
            "legitimate_segment": legit_stats,
            "fraudulent_segment": fraud_stats
        },
        "risk_insights": {
            "top_risk_hours": [f"{h:02d}:00" for h in top_risk_hours],
            "observation": "Fraudulent transactions demonstrate significantly higher standard deviation and mean amount compared to legitimate transactions."
        }
    })

@app.route('/api/chart/<chart_type>', methods=['GET'])
def get_chart(chart_type):
    """
    Feature 2: Matplotlib Chart Generator
    Generates dynamic statistical visualization and returns as a PNG stream.
    """
    global cached_df
    if cached_df is None:
        load_dataset()
        if cached_df is None:
            return jsonify({"status": "error", "message": "Dataset unavailable"}), 503

    df = cached_df.copy()
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.patch.set_facecolor('#070d1c')
    ax.set_facecolor('#0b1426')

    if chart_type == 'distribution':
        legit = df[df['fraud'] == 0]['amount'].dropna()
        fraud = df[df['fraud'] == 1]['amount'].dropna()
        ax.boxplot([legit, fraud], tick_labels=['Legitimate (0)', 'Fraud (1)'], patch_artist=True,
                   boxprops=dict(facecolor='#00f0ff', color='#00f0ff'),
                   medianprops=dict(color='#ff0055', linewidth=2),
                   whiskerprops=dict(color='#3b82f6'),
                   capprops=dict(color='#3b82f6'),
                   flierprops=dict(markeredgecolor='#f59e0b', markersize=3))
        ax.set_ylabel('Transaction Amount (INR)', color='#e2e8f0')
        ax.set_title('Fraud Status vs. Transaction Amount Distribution (Boxplot)', color='#00f0ff', pad=12)
        ax.grid(True, linestyle='--', alpha=0.2, color='#3b82f6')
    else:
        # Default chart: Hourly Fraud Counts
        hourly = df.groupby('transaction_hour')['fraud'].sum()
        ax.bar(hourly.index, hourly.values, color='#ff0055', edgecolor='#00f0ff', alpha=0.85)
        ax.set_xlabel('Transaction Hour (0 - 23)', color='#e2e8f0')
        ax.set_ylabel('Total Fraud Incidents', color='#e2e8f0')
        ax.set_title('Fraudulent Transaction Frequency by Hour', color='#00f0ff', pad=12)
        ax.grid(True, linestyle='--', alpha=0.2, color='#3b82f6')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

# ===========================================================================
# FEATURE THREE -> SUSPICIOUS TRANSACTION EXPLORER
# Allows querying, filtering, and investigating anomalous records from dataset
# ===========================================================================
@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """
    Feature 3: Suspicious Transaction Explorer
    Returns paginated, searchable transactions with anomaly scores and Z-scores.
    """
    global cached_df
    if cached_df is None:
        load_dataset()
        if cached_df is None:
            return jsonify({"status": "error", "message": "Dataset unavailable"}), 503

    df = cached_df.copy()

    # Calculate Z-score for outlier identification
    mean_val = df['amount'].mean()
    std_val = df['amount'].std()
    df['z_score'] = ((df['amount'] - mean_val) / std_val).round(2)

    # Filter parameters
    status_filter = request.args.get('filter', 'fraud')  # 'fraud', 'all', 'high_risk'
    search_query = request.args.get('search', '').strip()
    limit = min(int(request.args.get('limit', 15)), 50)

    filtered_df = df.copy()

    if status_filter == 'fraud':
        filtered_df = filtered_df[filtered_df['fraud'] == 1]
    elif status_filter == 'high_risk':
        filtered_df = filtered_df[(filtered_df['failed_attempts'] >= 2) | (filtered_df['z_score'] > 2.5)]

    if search_query:
        filtered_df = filtered_df[filtered_df['transaction_id'].str.contains(search_query, case=False, na=False)]

    # Sort to show most unusual or fraudulent cases first
    sorted_df = filtered_df.sort_values(by=['fraud', 'z_score'], ascending=[False, False]).head(limit)

    records = []
    for _, row in sorted_df.iterrows():
        records.append({
            "transaction_id": str(row['transaction_id']),
            "amount": float(row['amount']) if pd.notnull(row['amount']) else 0.0,
            "transaction_hour": int(row['transaction_hour']),
            "customer_age": int(row['customer_age']) if pd.notnull(row['customer_age']) else 30,
            "previous_transactions": int(row['previous_transactions']) if pd.notnull(row['previous_transactions']) else 0,
            "failed_attempts": int(row['failed_attempts']),
            "new_device": int(row['new_device']),
            "location_changed": int(row['location_changed']),
            "account_age_days": int(row['account_age_days']),
            "is_fraud": bool(row['fraud'] == 1),
            "z_score": float(row['z_score']) if pd.notnull(row['z_score']) else 0.0
        })

    return jsonify({
        "status": "success",
        "feature": "Feature 3: Suspicious Transaction Explorer",
        "total_matching": len(filtered_df),
        "count_returned": len(records),
        "filter_applied": status_filter,
        "transactions": records
    })

# ===========================================================================
# SERVER RUNNER
# ===========================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 75)
    print("  FRAUDSHIELD AI // BACKEND INITIALIZED ACCORDING TO NOTES.CSS")
    print("  - Feature 1: AI Transaction Analyser  (POST /predict)")
    print("  - Feature 2: Analytics Dashboard      (GET  /api/analytics, GET /api/chart)")
    print("  - Feature 3: Transaction Explorer     (GET  /api/transactions)")
    print("=" * 75)

    try:
        print(f"  Live Console: http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError:
        port = 5001
        print(f"  [PORT 5000 BUSY] Redirecting to port {port}")
        print(f"  Live Console: http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)
