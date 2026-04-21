# risk_scoring.py
# Step 4: Customer-level risk scoring — High / Medium / Low

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("data", exist_ok=True)

# ── Load predictions from Day 1 ───────────────────────────────────────────────
print("Loading data...")
merged      = pd.read_csv("data/merged_data.csv")
predictions = pd.read_csv("data/fraud_predictions.csv")

# Merge fraud probability back into master data
df = merged.merge(predictions[["transaction_id", "fraud_probability"]], 
                  on="transaction_id", how="left")

print(f"✅ Loaded {len(df)} transactions for {df['customer_id'].nunique()} customers")

# ── CUSTOMER RISK SCORE CALCULATION ───────────────────────────────────────────
print("\nCalculating customer risk scores...")

customer_risk = df.groupby("customer_id").agg(
    total_transactions    = ("transaction_id",    "count"),
    avg_fraud_probability = ("fraud_probability", "mean"),
    max_fraud_probability = ("fraud_probability", "max"),
    total_fraud_txns      = ("fraud_flag",        "sum"),
    avg_transaction_amt   = ("transaction_amount","mean"),
    max_transaction_amt   = ("transaction_amount","max"),
    avg_balance           = ("balance",           "mean"),
    high_risk_txns        = ("is_high_risk_merchant", "sum"),
    night_txns            = ("is_night_transaction",  "sum"),
    location_deviations   = ("location_deviation",    "sum"),
).reset_index()

# ── COMPOSITE RISK SCORE (weighted formula) ────────────────────────────────────
# Each factor contributes differently to overall risk
customer_risk["fraud_rate"] = (
    customer_risk["total_fraud_txns"] / customer_risk["total_transactions"]
)

customer_risk["risk_score"] = (
    customer_risk["avg_fraud_probability"]  * 40 +   # 40% weight
    customer_risk["fraud_rate"]             * 25 +   # 25% weight
    customer_risk["max_fraud_probability"]  * 20 +   # 20% weight
    (customer_risk["high_risk_txns"] / 
     customer_risk["total_transactions"])   * 10 +   # 10% weight
    (customer_risk["location_deviations"] / 
     customer_risk["total_transactions"])   *  5     #  5% weight
)

# Normalize to 0–100 scale
min_score = customer_risk["risk_score"].min()
max_score = customer_risk["risk_score"].max()
customer_risk["risk_score_normalized"] = (
    (customer_risk["risk_score"] - min_score) / (max_score - min_score) * 100
).round(2)

# ── ASSIGN RISK CATEGORY ──────────────────────────────────────────────────────
def assign_risk_category(score):
    if score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    else:
        return "Low"

customer_risk["risk_category"] = customer_risk["risk_score_normalized"].apply(
    assign_risk_category
)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("CUSTOMER RISK SCORING RESULTS")
print("="*55)

distribution = customer_risk["risk_category"].value_counts()
print(f"\nRisk Distribution:")
print(f"  🔴 High Risk   : {distribution.get('High',   0)} customers")
print(f"  🟡 Medium Risk : {distribution.get('Medium', 0)} customers")
print(f"  🟢 Low Risk    : {distribution.get('Low',    0)} customers")

print(f"\nTop 10 Highest Risk Customers:")
top10 = customer_risk.nlargest(10, "risk_score_normalized")[
    ["customer_id", "risk_score_normalized", "risk_category",
     "total_fraud_txns", "avg_fraud_probability", "high_risk_txns"]
]
print(top10.to_string(index=False))

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
output_cols = ["customer_id", "risk_score_normalized", "risk_category",
               "total_transactions", "total_fraud_txns", "fraud_rate",
               "avg_fraud_probability", "high_risk_txns", "location_deviations"]

customer_risk[output_cols].to_csv("data/customer_risk_scores.csv", index=False)
print("\n✅ Risk scores saved → data/customer_risk_scores.csv")

# ── CHART: Risk Distribution Pie Chart ────────────────────────────────────────
colors = ["#e74c3c", "#f39c12", "#2ecc71"]
labels = ["High Risk", "Medium Risk", "Low Risk"]
sizes  = [
    distribution.get("High",   0),
    distribution.get("Medium", 0),
    distribution.get("Low",    0),
]

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, textprops={"fontsize": 12})
plt.title("Customer Risk Distribution\nHigh / Medium / Low",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/customer_risk_distribution.png", dpi=150)
plt.close()
print("✅ Chart saved → data/customer_risk_distribution.png")

print("\n🎉 Step 4 Complete — risk_scoring.py finished!")