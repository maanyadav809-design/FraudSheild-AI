"""
=============================================================================
FRAUDSHIELD AI // FLASK BACKEND & MACHINE LEARNING INFERENCE SERVICE
=============================================================================
Course / Project Specifications:
Part 4 - Start the backend
The backend provides four exact endpoints:
1. GET  /api/health        -> Check whether the API and model are available
2. GET  /api/stats         -> Return dataset statistics
3. GET  /api/model-metrics  -> Return held-out evaluation metrics
4. POST /api/predict       -> Return a prediction for a submitted transaction

Additional Features:
- Route: GET /              -> Renders AI Dashboard (templates/index.html)
- Route: GET /api/chart/<t> -> Generates Matplotlib boxplot & distribution charts
- Route: GET /api/transactions -> Suspicious Transaction Explorer
=============================================================================
"""

import os
import sys
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# Set Matplotlib to non-interactive Agg backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows consoles
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
# Global In-Memory Model & Dataset Cache
# ---------------------------------------------------------------------------
ml_pipeline = None
cached_df = None

def load_fraud_model():
    """Loads the serialized Logistic Regression pipeline from models/fraud_model.joblib."""
    global ml_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            ml_pipeline = joblib.load(MODEL_PATH)
            print(f"[*] [MODEL LOADED] Logistic Regression pipeline ready from: {MODEL_PATH}")
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
# ROUTE: WEB DASHBOARD VIEW
# ===========================================================================
@app.route('/')
def home():
    """Renders the primary Cyber-Robotic UI Console."""
    return render_template('index.html')


# ===========================================================================
# 1. ENDPOINT: GET /api/health
# Purpose: Check whether the API and model are available
# ===========================================================================
@app.route('/api/health', methods=['GET'])
def health():
    """Checks API availability and whether the trained ML model is loaded."""
    is_ready = ml_pipeline is not None
    return jsonify({
        "status": "ok" if is_ready else "error",
        "api_available": True,
        "model_available": is_ready,
        "model_type": "LogisticRegression" if is_ready else None,
        "service": "FraudShield AI Sentinel Core"
    })


# ===========================================================================
# 2. ENDPOINT: GET /api/stats
# Purpose: Return dataset statistics
# ===========================================================================
@app.route('/api/stats', methods=['GET'])
@app.route('/api/analytics', methods=['GET'])
def get_stats():
    """Computes and returns comprehensive dataset statistics using Pandas and NumPy."""
    global cached_df
    if cached_df is None:
        load_dataset()
        if cached_df is None:
            return jsonify({"status": "error", "message": "Dataset unavailable"}), 503

    df = cached_df.copy()
    amount_col = df['amount'].dropna()

    total_txns = len(df)
    fraud_counts = df['fraud'].value_counts()
    fraud_count = int(fraud_counts.get(1, 0))
    legit_count = int(fraud_counts.get(0, 0))
    fraud_percentage = round((fraud_count / total_txns) * 100, 2)

    mean_amt = float(amount_col.mean())
    median_amt = float(amount_col.median())
    std_amt = float(amount_col.std())
    min_amt = float(amount_col.min())
    max_amt = float(amount_col.max())
    q1 = float(amount_col.quantile(0.25))
    q3 = float(amount_col.quantile(0.75))
    iqr = q3 - q1

    # Breakdown by fraud segment
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

    return jsonify({
        "status": "success",
        "total_transactions": total_txns,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_percentage": fraud_percentage,
        "amount_mean": round(mean_amt, 2),
        "amount_median": round(median_amt, 2),
        "amount_std": round(std_amt, 2),
        "amount_min": round(min_amt, 2),
        "amount_max": round(max_amt, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "dataset_summary": {
            "total_transactions": total_txns,
            "legitimate_count": legit_count,
            "fraudulent_count": fraud_count,
            "fraud_rate_percent": fraud_percentage
        },
        "segment_comparison": {
            "legitimate_segment": legit_stats,
            "fraudulent_segment": fraud_stats
        }
    })


# ===========================================================================
# 3. ENDPOINT: GET /api/model-metrics
# Purpose: Return held-out evaluation metrics
# ===========================================================================
@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    """Returns the held-out test evaluation metrics for the trained Logistic Regression model."""
    global ml_pipeline
    if ml_pipeline is None:
        load_fraud_model()
        if ml_pipeline is None:
            return jsonify({"status": "error", "message": "Model not available"}), 503

    eval_metrics = ml_pipeline.get('evaluation_metrics', {})
    coefficients = ml_pipeline.get('coefficients', {})

    return jsonify({
        "status": "success",
        "model_type": "LogisticRegression",
        "accuracy": round(float(eval_metrics.get('accuracy', 0.0)), 4),
        "precision": round(float(eval_metrics.get('precision', 0.0)), 4),
        "recall": round(float(eval_metrics.get('recall', 0.0)), 4),
        "f1_score": round(float(eval_metrics.get('f1_score', 0.0)), 4),
        "roc_auc": round(float(eval_metrics.get('roc_auc', 0.0)), 4),
        "confusion_matrix": eval_metrics.get('confusion_matrix', []),
        "coefficients": coefficients,
        "intercept": ml_pipeline.get('intercept', 0.0)
    })


# ===========================================================================
# 4. ENDPOINT: POST /api/predict
# Purpose: Return a prediction for a submitted transaction
# Tested with Postman payload from Step 7:
# {
#   "amount": 78500,
#   "transaction_hour": 2,
#   "customer_age": 29,
#   "previous_transactions": 1,
#   "failed_attempts": 4,
#   "new_device": true,
#   "location_changed": true,
#   "account_age_days": 80
# }
# ===========================================================================
@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
@app.route('/api/analyze', methods=['POST'])
def predict_transaction():
    """
    Receives transaction vector, passes it through StandardScaler and
    LogisticRegression model, and returns the prediction and probability.
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

        # Step 1: Extract and convert inputs into expected feature structure
        impute_vals = ml_pipeline.get('impute_values', {})
        amount = float(data.get('amount', impute_vals.get('amount', 1191.19)))
        hour = int(data.get('transaction_hour', data.get('hour', 12)))
        age = int(data.get('customer_age', impute_vals.get('customer_age', 34)))
        prev_txns = int(data.get('previous_transactions', impute_vals.get('previous_transactions', 5)))
        failed = int(data.get('failed_attempts', 0))

        # Support both boolean (true/false) and integer (1/0)
        new_dev = 1 if data.get('new_device') in [1, True, '1', 'true', 'True'] else 0
        loc_changed = 1 if data.get('location_changed') in [1, True, '1', 'true', 'True'] else 0
        acc_age = int(data.get('account_age_days', data.get('account_age', 180)))
        tx_id = str(data.get('transaction_id', data.get('txId', 'TXN-API-SUBMITTED')))

        # Step 2: Format features into DataFrame matching trained model order
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
        X_input = pd.DataFrame([feature_dict])[feature_cols]

        # Step 3: Standardize features using the saved pipeline scaler
        scaler = ml_pipeline['scaler']
        model = ml_pipeline['model']
        X_scaled = scaler.transform(X_input)

        # Step 4: Run inference through the saved Logistic Regression model
        pred_class = int(model.predict(X_scaled)[0])
        pred_prob = float(model.predict_proba(X_scaled)[0, 1])
        risk_score_pct = round(pred_prob * 100, 2)

        # Step 5: Risk Level & Verdict Details
        if pred_class == 1 or risk_score_pct >= 65.0:
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

        # Step 6: Explainable AI Feature Attribution Breakdown
        coefficients = ml_pipeline['coefficients']
        scaled_features = X_scaled[0]
        attributions = []

        for feat_name, s_val in zip(feature_cols, scaled_features):
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

        # Step 7: Return exact prediction and probability matching requirements
        return jsonify({
            "status": "success",
            "prediction": pred_class,
            "probability": round(pred_prob, 6),
            "fraud_probability": round(pred_prob, 6),
            "is_fraud": bool(pred_class == 1),
            "risk_score_percent": risk_score_pct,
            "risk_level": risk_level,
            "verdict_badge": verdict_badge,
            "verdict_title": verdict_title,
            "verdict_description": verdict_desc,
            "recommendation": rec_action,
            "feature_attributions": attributions,
            "inputs_evaluated": feature_dict,
            "transaction_id": tx_id
        })

    except Exception as e:
        print(f"[!] Inference Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===========================================================================
# ADDITIONAL FEATURES (Matplotlib Chart & Transaction Explorer)
# ===========================================================================
@app.route('/api/chart/<chart_type>', methods=['GET'])
def get_chart(chart_type):
    """Generates dynamic dark-themed Matplotlib visualization and returns as a PNG stream."""
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

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Returns paginated, searchable transactions with anomaly scores and Z-scores."""
    global cached_df
    if cached_df is None:
        load_dataset()
        if cached_df is None:
            return jsonify({"status": "error", "message": "Dataset unavailable"}), 503

    df = cached_df.copy()
    mean_val = df['amount'].mean()
    std_val = df['amount'].std()
    df['z_score'] = ((df['amount'] - mean_val) / std_val).round(2)

    status_filter = request.args.get('filter', 'fraud')
    search_query = request.args.get('search', '').strip()
    limit = min(int(request.args.get('limit', 15)), 50)

    filtered_df = df.copy()
    if status_filter == 'fraud':
        filtered_df = filtered_df[filtered_df['fraud'] == 1]
    elif status_filter == 'high_risk':
        filtered_df = filtered_df[(filtered_df['failed_attempts'] >= 2) | (filtered_df['z_score'] > 2.5)]

    if search_query:
        filtered_df = filtered_df[filtered_df['transaction_id'].str.contains(search_query, case=False, na=False)]

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
    print("  FRAUDSHIELD AI // FLASK BACKEND RUNNING")
    print("  Curriculum Endpoints Available:")
    print("    1. GET  /api/health        - Check API and model availability")
    print("    2. GET  /api/stats         - Return dataset statistics")
    print("    3. GET  /api/model-metrics - Return held-out evaluation metrics")
    print("    4. POST /api/predict       - Return prediction for submitted transaction")
    print("=" * 75)

    try:
        print(f"  Live Dashboard: http://127.0.0.1:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError:
        port = 5001
        print(f"  [PORT 5000 BUSY] Redirecting to port {port}")
        print(f"  Live Dashboard: http://127.0.0.1:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)
