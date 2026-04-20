# fraud_detection.py
# Step 3: Feature engineering + Random Forest fraud detection model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

os.makedirs("data", exist_ok=True)

# ── Load all 4 datasets ────────────────────────────────────────────────────────
print("Loading datasets...")
customers    = pd.read_csv("data/customers.csv")
accounts     = pd.read_csv("data/accounts.csv")
transactions = pd.read_csv("data/transactions.csv")
fraud_labels = pd.read_csv("data/fraud_labels.csv")

# Parse dates
transactions["timestamp"]          = pd.to_datetime(transactions["timestamp"])
accounts["account_open_date"]      = pd.to_datetime(accounts["account_open_date"])

# ── Merge into one master dataframe ───────────────────────────────────────────
df = transactions.merge(fraud_labels, on="transaction_id")
df = df.merge(accounts, on="account_id")
df = df.merge(customers, on="customer_id")
print(f"✅ Master dataframe: {df.shape[0]} rows × {df.shape[1]} columns")

# ── FEATURE ENGINEERING (11 features) ─────────────────────────────────────────
print("\nEngineering features...")

# Time-based features
df["hour_of_day"]          = df["timestamp"].dt.hour
df["is_night_transaction"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
df["is_weekend"]           = (df["timestamp"].dt.dayofweek >= 5).astype(int)

# Account behaviour features
txn_stats = df.groupby("account_id")["transaction_amount"].agg(
    transaction_frequency="count",
    avg_spending="mean",
    std_spending="std"
).reset_index()
txn_stats["std_spending"] = txn_stats["std_spending"].fillna(0)
df = df.merge(txn_stats, on="account_id")

# Amount Z-score: how unusual is this transaction for THIS account?
df["amount_zscore"] = df.apply(
    lambda row: (row["transaction_amount"] - row["avg_spending"]) / row["std_spending"]
    if row["std_spending"] > 0 else 0,
    axis=1
)

# Location deviation: transaction not in customer's home city?
df["location_deviation"] = (df["location"] != df["city"]).astype(int)

# High-risk merchant flag
df["is_high_risk_merchant"] = df["merchant_category"].isin(["Gambling", "Jewelry"]).astype(int)

# Financial ratio features
df["amount_to_balance_ratio"]     = df["transaction_amount"] / (df["balance"] + 1)
df["txn_to_monthly_income_ratio"] = df["transaction_amount"] / (df["annual_income"] / 12)

# Account age in days at time of transaction
df["account_age_days"] = (df["timestamp"] - df["account_open_date"]).dt.days

# Save merged data
df.to_csv("data/merged_data.csv", index=False)
print("✅ Feature engineering complete — 11 features created")
print("\nFeature summary:")
features = ["hour_of_day", "is_night_transaction", "is_weekend",
            "transaction_frequency", "avg_spending", "amount_zscore",
            "location_deviation", "is_high_risk_merchant",
            "amount_to_balance_ratio", "txn_to_monthly_income_ratio",
            "account_age_days"]
print(df[features].describe().round(3).to_string())

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
X = df[features]
y = df["fraud_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Fraud in train: {y_train.sum()} | Fraud in test: {y_test.sum()}")

# ── RANDOM FOREST MODEL ───────────────────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",   # handles imbalanced fraud data
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── EVALUATION ────────────────────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# ── SAVE PREDICTIONS ──────────────────────────────────────────────────────────
df["fraud_probability"] = model.predict_proba(X)[:, 1]
predictions = df[["transaction_id", "fraud_probability", "fraud_flag"]].copy()
predictions.to_csv("data/fraud_predictions.csv", index=False)
print("✅ Predictions saved → data/fraud_predictions.csv")
print("\nSample predictions:")
print(predictions.sort_values("fraud_probability", ascending=False).head(10).to_string(index=False))

# ── CHART 1: Feature Importance ───────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(10, 7))
importances.plot(kind="barh", color="steelblue")
plt.title("Feature Importance — Random Forest Fraud Detection", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=150)
plt.close()
print("✅ Chart saved → data/feature_importance.png")

# ── CHART 2: Confusion Matrix ─────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Legit", "Predicted Fraud"],
            yticklabels=["Actual Legit", "Actual Fraud"])
plt.title("Confusion Matrix — Fraud Detection Model", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=150)
plt.close()
print("✅ Chart saved → data/confusion_matrix.png")

print("\n🎉 Day 1 Complete! fraud_detection.py finished successfully.")