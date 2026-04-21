# insights.py
# Step 5: Business insights + 6-panel dashboard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("data", exist_ok=True)

# ── Load all outputs ───────────────────────────────────────────────────────────
print("Loading all data...")
merged      = pd.read_csv("data/merged_data.csv")
predictions = pd.read_csv("data/fraud_predictions.csv")
risk_scores = pd.read_csv("data/customer_risk_scores.csv")

df = merged.merge(
    predictions[["transaction_id", "fraud_probability"]],
    on="transaction_id", how="left"
)
print(f"✅ Loaded {len(df)} transactions")

# ── BUSINESS INSIGHTS ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*60)

# Insight 1: Fraud by merchant
fraud_by_merchant = df.groupby("merchant_category").agg(
    total=("fraud_flag","count"),
    fraud=("fraud_flag","sum")
).reset_index()
fraud_by_merchant["fraud_rate"] = (
    fraud_by_merchant["fraud"] / fraud_by_merchant["total"] * 100
).round(2)
fraud_by_merchant = fraud_by_merchant.sort_values("fraud_rate", ascending=False)

print("\n📊 Insight 1 — Fraud Rate by Merchant Category:")
print(fraud_by_merchant[["merchant_category","total","fraud","fraud_rate"]].to_string(index=False))

# Insight 2: Night vs Day
df["time_period"] = df["is_night_transaction"].map({1:"Night (10PM-5AM)", 0:"Day"})
night_fraud = df.groupby("time_period")["fraud_flag"].agg(["count","sum","mean"]).reset_index()
night_fraud["fraud_rate_%"] = (night_fraud["mean"] * 100).round(2)
print("\n📊 Insight 2 — Night vs Day Fraud:")
print(night_fraud[["time_period","count","sum","fraud_rate_%"]].to_string(index=False))

# Insight 3: Location deviation impact
loc_fraud = df.groupby("location_deviation")["fraud_flag"].agg(["count","sum","mean"]).reset_index()
loc_fraud["label"] = loc_fraud["location_deviation"].map(
    {0:"Same City", 1:"Different City"}
)
loc_fraud["fraud_rate_%"] = (loc_fraud["mean"] * 100).round(2)
print("\n📊 Insight 3 — Location Deviation Impact:")
print(loc_fraud[["label","count","sum","fraud_rate_%"]].to_string(index=False))

# Insight 4: High value fraud
high_value = df[df["transaction_amount"] > 50000]
print(f"\n📊 Insight 4 — High Value Transactions (>₹50,000):")
print(f"   Total   : {len(high_value)}")
print(f"   Fraud   : {high_value['fraud_flag'].sum()}")
print(f"   Rate    : {high_value['fraud_flag'].mean()*100:.2f}%")

# Insight 5: Risk category summary
risk_summary = risk_scores.groupby("risk_category").agg(
    customers       = ("customer_id",           "count"),
    avg_risk_score  = ("risk_score_normalized", "mean"),
    avg_fraud_txns  = ("total_fraud_txns",      "mean"),
).reset_index()
print("\n📊 Insight 5 — Risk Category Summary:")
print(risk_summary.to_string(index=False))

# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TOP RECOMMENDATIONS FOR THE BANK")
print("="*60)
print("""
1. 🚨 FLAG Gambling & Jewelry transactions for manual review
   → These merchant categories show the highest fraud rates

2. 🌙 MONITOR night transactions (10PM–5AM) more closely
   → Fraud rate is significantly higher at night

3. 📍 ALERT on cross-city transactions immediately
   → Location deviation is a strong fraud signal

4. 💰 BLOCK or HOLD transactions above ₹50,000 for verification
   → High-value transactions carry elevated fraud risk

5. 👥 ASSIGN relationship managers to High Risk customers
   → Proactive monitoring can prevent fraud before it happens
""")

# ── 6-PANEL DASHBOARD ─────────────────────────────────────────────────────────
print("Building dashboard...")
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Banking Fraud Detection — Business Insights Dashboard",
             fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Fraud rate by merchant
ax1 = fig.add_subplot(gs[0, 0])
colors_merch = ["#e74c3c" if r > 15 else "#f39c12" if r > 8 else "#2ecc71"
                for r in fraud_by_merchant["fraud_rate"]]
ax1.barh(fraud_by_merchant["merchant_category"],
         fraud_by_merchant["fraud_rate"], color=colors_merch)
ax1.set_title("Fraud Rate by Merchant Category (%)", fontweight="bold")
ax1.set_xlabel("Fraud Rate (%)")
ax1.axvline(x=df["fraud_flag"].mean()*100, color="black",
            linestyle="--", linewidth=1, label="Avg fraud rate")
ax1.legend(fontsize=8)

# Panel 2: Transaction amount distribution
ax2 = fig.add_subplot(gs[0, 1])
fraud_amounts  = df[df["fraud_flag"] == 1]["transaction_amount"]
legit_amounts  = df[df["fraud_flag"] == 0]["transaction_amount"]
ax2.hist(legit_amounts,  bins=40, alpha=0.6, color="#2ecc71", label="Legitimate")
ax2.hist(fraud_amounts,  bins=40, alpha=0.7, color="#e74c3c", label="Fraud")
ax2.set_title("Transaction Amount: Fraud vs Legitimate", fontweight="bold")
ax2.set_xlabel("Transaction Amount (₹)")
ax2.set_ylabel("Count")
ax2.legend()

# Panel 3: Fraud by hour of day
ax3 = fig.add_subplot(gs[1, 0])
hourly = df.groupby("hour_of_day")["fraud_flag"].mean() * 100
ax3.plot(hourly.index, hourly.values, color="#e74c3c",
         linewidth=2, marker="o", markersize=4)
ax3.fill_between(hourly.index, hourly.values, alpha=0.2, color="#e74c3c")
ax3.set_title("Fraud Rate by Hour of Day", fontweight="bold")
ax3.set_xlabel("Hour (0 = Midnight)")
ax3.set_ylabel("Fraud Rate (%)")
ax3.axhspan(0, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 20,
            xmin=22/24, alpha=0.1, color="red")

# Panel 4: Customer risk distribution
ax4 = fig.add_subplot(gs[1, 1])
risk_dist  = risk_scores["risk_category"].value_counts()
risk_colors = {"High":"#e74c3c", "Medium":"#f39c12", "Low":"#2ecc71"}
bar_colors  = [risk_colors[cat] for cat in risk_dist.index]
ax4.bar(risk_dist.index, risk_dist.values, color=bar_colors, edgecolor="white")
ax4.set_title("Customer Risk Distribution", fontweight="bold")
ax4.set_ylabel("Number of Customers")
for i, (cat, val) in enumerate(zip(risk_dist.index, risk_dist.values)):
    ax4.text(i, val + 1, str(val), ha="center", fontweight="bold")

# Panel 5: Location deviation fraud impact
ax5 = fig.add_subplot(gs[2, 0])
loc_labels = ["Same City", "Different City"]
loc_rates  = loc_fraud.sort_values("location_deviation")["fraud_rate_%"].values
ax5.bar(loc_labels, loc_rates, color=["#2ecc71","#e74c3c"], width=0.5)
ax5.set_title("Fraud Rate: Same City vs Different City", fontweight="bold")
ax5.set_ylabel("Fraud Rate (%)")
for i, v in enumerate(loc_rates):
    ax5.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")

# Panel 6: Top 10 risky customers
ax6 = fig.add_subplot(gs[2, 1])
top10 = risk_scores.nlargest(10, "risk_score_normalized")
bar_c = ["#e74c3c" if c == "High" else "#f39c12"
         for c in top10["risk_category"]]
ax6.barh(top10["customer_id"], top10["risk_score_normalized"], color=bar_c)
ax6.set_title("Top 10 Highest Risk Customers", fontweight="bold")
ax6.set_xlabel("Risk Score (0–100)")
ax6.invert_yaxis()

plt.savefig("data/insights_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dashboard saved → data/insights_dashboard.png")

print("\n🎉 Step 5 Complete — insights.py finished!")
print("\n" + "="*60)
print("✅ ALL 5 STEPS COMPLETE — PROJECT DONE!")
print("="*60)