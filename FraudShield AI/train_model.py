"""
=============================================================================
FRAUDSHIELD AI // LOGISTIC REGRESSION MODEL TRAINING PIPELINE
=============================================================================
Course / Project: Statistics for AI — Fraud Detection
Model: Logistic Regression Classifier for Binary Fraud Classification (0=Legit, 1=Fraud)

Steps:
1. Load dataset & inspect missing values
2. Clean data & handle missing values via median imputation
3. Separate input features (X) and target label (y)
4. Split dataset into train and test sets (train_test_split, 80/20, stratify)
5. Standardize numerical features & train Logistic Regression model
6. Evaluate model: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
7. Extract feature coefficients (Explainable AI / Feature Importance)
8. Serialize and save the model pipeline to models/fraud_model.joblib
9. Validate against the student benchmark challenge (₹52K Midnight Attack)
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Support UTF-8 encoding in Windows consoles
if sys.platform == 'win32' and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def train_fraud_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'AI_Fraud_Detection_Transactions.csv')
    models_dir = os.path.join(base_dir, 'models')
    model_save_path = os.path.join(models_dir, 'fraud_model.joblib')

    os.makedirs(models_dir, exist_ok=True)

    print("=" * 75)
    print("  FRAUDSHIELD AI // TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 75)

    # ---------------------------------------------------------
    # STEP 1: Load and Inspect the Data
    # ---------------------------------------------------------
    print(f"\n[STEP 1] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"[*] Dataset Shape: {df.shape[0]} records, {df.shape[1]} columns")

    # ---------------------------------------------------------
    # STEP 2: Data Cleaning & Missing Value Imputation
    # ---------------------------------------------------------
    print("\n[STEP 2] Cleaning data & handling missing values...")
    missing_before = df.isnull().sum()
    print("[*] Missing values detected before cleaning:")
    for col, count in missing_before.items():
        if count > 0:
            print(f"    - {col}: {count} missing values")

    # Compute and store median values for production inference imputation
    impute_values = {
        'amount': float(df['amount'].median()),
        'customer_age': float(df['customer_age'].median()),
        'previous_transactions': float(df['previous_transactions'].median())
    }

    df['amount'] = df['amount'].fillna(impute_values['amount'])
    df['customer_age'] = df['customer_age'].fillna(impute_values['customer_age'])
    df['previous_transactions'] = df['previous_transactions'].fillna(impute_values['previous_transactions'])

    print(f"[*] Imputed missing values using feature medians: {impute_values}")

    # ---------------------------------------------------------
    # STEP 3: Separate Inputs (X) and Target (y)
    # ---------------------------------------------------------
    feature_cols = [
        'amount',
        'transaction_hour',
        'customer_age',
        'previous_transactions',
        'failed_attempts',
        'new_device',
        'location_changed',
        'account_age_days'
    ]

    target_col = 'fraud'

    X = df[feature_cols]
    y = df[target_col]

    fraud_counts = y.value_counts()
    fraud_pct = y.value_counts(normalize=True) * 100
    print(f"\n[STEP 3] Target Distribution: 0 (Legitimate)={fraud_counts[0]} ({fraud_pct[0]:.2f}%), 1 (Fraud)={fraud_counts[1]} ({fraud_pct[1]:.2f}%)")

    # ---------------------------------------------------------
    # STEP 4: Train / Test Split
    # ---------------------------------------------------------
    print("\n[STEP 4] Splitting data into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"[*] Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

    # ---------------------------------------------------------
    # STEP 5: Feature Scaling & Model Training
    # ---------------------------------------------------------
    print("\n[STEP 5] Standardizing features and fitting Logistic Regression Classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use class_weight='balanced' to handle the ~1.78% fraud imbalance effectively
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    print("[*] Model convergence reached. Parameters learned.")

    # ---------------------------------------------------------
    # STEP 6: Model Evaluation on Test Set
    # ---------------------------------------------------------
    print("\n[STEP 6] Evaluating Model on Unseen Test Set (20% Holdout):")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"    - Accuracy:  {acc * 100:.2f}%")
    print(f"    - Precision: {prec * 100:.2f}%")
    print(f"    - Recall:    {rec * 100:.2f}% (High sensitivity to catch threats)")
    print(f"    - F1-Score:  {f1:.4f}")
    print(f"    - ROC-AUC:   {auc:.4f}")
    print("\n[*] Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print("\n[*] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraud (1)']))

    # ---------------------------------------------------------
    # STEP 7: Feature Importance / Learned Weights
    # ---------------------------------------------------------
    print("\n[STEP 7] Model Feature Weights (Coefficients):")
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': coefficients,
        'Absolute_Impact': np.abs(coefficients)
    }).sort_values(by='Absolute_Impact', ascending=False)

    for _, row in feature_importance.iterrows():
        direction = "Increases Risk (+)" if row['Coefficient'] > 0 else "Lowers Risk (-)"
        print(f"    - {row['Feature']:<24}: {row['Coefficient']:+.4f} ({direction})")

    # ---------------------------------------------------------
    # STEP 8: Save Model Pipeline to Disk
    # ---------------------------------------------------------
    model_payload = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_cols,
        'impute_values': impute_values,
        'evaluation_metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'confusion_matrix': cm.tolist()
        },
        'coefficients': dict(zip(feature_cols, [float(c) for c in coefficients])),
        'intercept': float(model.intercept_[0])
    }

    joblib.dump(model_payload, model_save_path)
    print(f"\n[STEP 8] Model and preprocessing pipeline saved successfully!")
    print(f"[*] Artifact Location: {model_save_path}")

    # ---------------------------------------------------------
    # STEP 9: Test on the Student Final Challenge Case
    # ---------------------------------------------------------
    print("\n[STEP 9] Validating on Student Project Challenge Scenario:")
    print("    - Amount: INR 52,000 | Hour: 03:00 (Night) | Age: 26")
    print("    - Failed Attempts: 5 | New Device: 1 | Location Changed: 1 | Account Age: 45d | Prev: 0")

    challenge_sample = pd.DataFrame([{
        'amount': 52000.0,
        'transaction_hour': 3,
        'customer_age': 26,
        'previous_transactions': 0,
        'failed_attempts': 5,
        'new_device': 1,
        'location_changed': 1,
        'account_age_days': 45
    }])[feature_cols]

    challenge_scaled = scaler.transform(challenge_sample)
    pred_prob = model.predict_proba(challenge_scaled)[0, 1]
    pred_class = model.predict(challenge_scaled)[0]

    print(f"\n[PREDICTION RESULT FOR BENCHMARK ATTACK]:")
    print(f"    - Predicted Class:       {pred_class} ({'FRAUD DETECTED' if pred_class == 1 else 'LEGITIMATE'})")
    print(f"    - Fraud Probability:     {pred_prob * 100:.2f}%")
    print(f"    - Verdict:               {'FLAGGED FOR IMMEDIATE QUARANTINE' if pred_prob > 0.5 else 'PASSED'}")
    print("=" * 75)

if __name__ == '__main__':
    train_fraud_model()
