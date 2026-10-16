"""
=============================================================================
FRAUDSHIELD AI // FLASK BACKEND & MACHINE LEARNING INFERENCE SERVICE
=============================================================================
Architecture:
- Framework: Flask
- ML Model: Logistic Regression Classifier (Serialized via Joblib)
- Artifact: models/fraud_model.joblib
- Endpoints:
    GET  /              -> Renders AI Dashboard (templates/index.html)
    GET  /api/health    -> Health check & model metadata
    POST /predict       -> Real-time ML Inference using Logistic Regression
    POST /api/analyze   -> Alias for /predict matching frontend telemetry schema
=============================================================================
"""

import os
import sys
import webbrowser
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Windows terminal UTF-8 support
if sys.platform == 'win32' and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Paths for templates and static folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.joblib')

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR,
    static_url_path='/static'
)

# ---------------------------------------------------------------------------
# Global Model Storage & Startup Loader
# ---------------------------------------------------------------------------
ml_pipeline = None

def load_fraud_model():
    global ml_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            ml_pipeline = joblib.load(MODEL_PATH)
            print(f"[*] [MODEL LOADED] Successfully loaded Logistic Regression model from: {MODEL_PATH}")
        except Exception as e:
            print(f"[!] [ERROR] Failed to load model artifact: {e}")
            ml_pipeline = None
    else:
        print(f"[!] [WARNING] Model artifact not found at {MODEL_PATH}.")
        print("[!] Please run 'python train_model.py' to train and serialize the model.")

load_fraud_model()

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    """Renders the futuristic robotic dashboard."""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Returns system status, active model type, and evaluation benchmarks."""
    is_ready = ml_pipeline is not None
    return jsonify({
        "status": "online" if is_ready else "degraded",
        "service": "FraudShield AI Sentinel Core",
        "model_loaded": is_ready,
        "model_type": "Logistic Regression Classifier" if is_ready else "Unavailable",
        "metrics": ml_pipeline.get('evaluation_metrics') if is_ready else None
    })

@app.route('/predict', methods=['POST'])
@app.route('/api/analyze', methods=['POST'])
def predict():
    """
    Real-time fraud prediction endpoint using Logistic Regression.
    Accepts JSON input matching dataset schema.
    """
    global ml_pipeline
    if ml_pipeline is None:
        load_fraud_model()
        if ml_pipeline is None:
            return jsonify({
                "status": "error",
                "message": "Model not trained yet. Run 'python train_model.py' first."
            }), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        # Extract features with robust fallbacks
        impute_vals = ml_pipeline.get('impute_values', {})
        amount = float(data.get('amount', impute_vals.get('amount', 1191.19)))
        hour = int(data.get('transaction_hour', data.get('hour', 12)))
        age = int(data.get('customer_age', impute_vals.get('customer_age', 34)))
        prev_txns = int(data.get('previous_transactions', impute_vals.get('previous_transactions', 5)))
        failed = int(data.get('failed_attempts', 0))

        # Handle boolean or integer for binary flags
        new_dev = 1 if data.get('new_device') in [1, True, '1', 'true', 'True'] else 0
        loc_changed = 1 if data.get('location_changed') in [1, True, '1', 'true', 'True'] else 0
        acc_age = int(data.get('account_age_days', data.get('account_age', 180)))
        tx_id = str(data.get('transaction_id', data.get('txId', 'TXN-00000')))

        # Build feature vector matching model training order
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

        # Standardize features using the fitted scaler
        scaler = ml_pipeline['scaler']
        model = ml_pipeline['model']
        X_scaled = scaler.transform(X_input)

        # Inference: Predict Class and Fraud Probability
        pred_class = int(model.predict(X_scaled)[0])
        pred_prob = float(model.predict_proba(X_scaled)[0, 1])
        risk_score_pct = round(pred_prob * 100, 2)

        # Determine Risk Level Category
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

        # Compute Explainable AI (XAI) feature attribution breakdown
        coefficients = ml_pipeline['coefficients']
        scaled_features = X_scaled[0]
        attributions = []

        for feat_name, scaled_val in zip(feature_cols, scaled_features):
            coef = coefficients.get(feat_name, 0.0)
            impact = coef * scaled_val  # log-odds contribution
            if abs(impact) > 0.05:
                direction = "danger" if impact > 0 else "safe"
                attributions.append({
                    "feature": feat_name,
                    "impact": round(impact, 3),
                    "direction": direction,
                    "weight_pct": min(round(abs(impact) * 22, 1), 95)
                })

        # Sort attributions by descending impact
        attributions.sort(key=lambda x: abs(x['impact']), reverse=True)

        response_payload = {
            "status": "success",
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
        }

        print(f"[*] Inference for {tx_id}: Score={risk_score_pct}% | Class={pred_class} | Status={risk_level}")
        return jsonify(response_payload)

    except Exception as e:
        print(f"[!] Inference Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 75)
    print("  FRAUDSHIELD AI // FLASK SERVER & AI INFERENCE CORE")
    print(f"  ML Model Status: {'Ready (Logistic Regression)' if ml_pipeline else 'Model file missing'}")
    print("=" * 75)

    try:
        print(f"  Live Dashboard: http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError:
        port = 5001
        print(f"  [PORT 5000 IN USE] Switching to port {port}")
        print(f"  Live Dashboard: http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=False)
