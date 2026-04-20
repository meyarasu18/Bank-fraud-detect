# generate_data.py
# Step 1: Generate synthetic banking datasets with realistic fraud patterns

import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# ── 1. CUSTOMERS ──────────────────────────────────────────────────────────────
n_customers = 500
cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Coimbatore"]
occupations = ["Engineer", "Doctor", "Teacher", "Businessman",
               "Lawyer", "Accountant", "Salesperson", "Manager"]

customers = pd.DataFrame({
    "customer_id":   [f"C{str(i).zfill(4)}" for i in range(1, n_customers + 1)],
    "age":           np.random.randint(22, 65, n_customers),
    "occupation":    np.random.choice(occupations, n_customers),
    "annual_income": np.random.randint(300000, 2000000, n_customers),
    "city":          np.random.choice(cities, n_customers),
})
customers.to_csv("data/customers.csv", index=False)
print(f"✅ Customers: {len(customers)} rows")

# ── 2. ACCOUNTS ───────────────────────────────────────────────────────────────
account_types = ["Savings", "Current", "Salary"]
accounts = pd.DataFrame({
    "account_id":        [f"A{str(i).zfill(5)}" for i in range(1, n_customers + 1)],
    "customer_id":       customers["customer_id"].values,
    "account_type":      np.random.choice(account_types, n_customers),
    "balance":           np.random.randint(5000, 500000, n_customers),
    "account_open_date": pd.to_datetime(
        np.random.choice(pd.date_range("2018-01-01", "2023-01-01"), n_customers)
    ),
})
accounts.to_csv("data/accounts.csv", index=False)
print(f"✅ Accounts:  {len(accounts)} rows")

# ── 3. TRANSACTIONS ───────────────────────────────────────────────────────────
n_transactions = 10000
merchant_categories = ["Groceries", "Electronics", "Clothing", "Restaurants",
                        "Travel", "Gambling", "Jewelry", "Healthcare", "Education"]
transaction_types = ["Debit", "Credit", "UPI", "Online"]

txn_account_ids = np.random.choice(accounts["account_id"], n_transactions)

# Map account → customer → city for location deviation feature
acc_to_cust = dict(zip(accounts["account_id"], accounts["customer_id"]))
cust_to_city = dict(zip(customers["customer_id"], customers["city"]))

txn_locations = []
for acc in txn_account_ids:
    home_city = cust_to_city[acc_to_cust[acc]]
    # 30% chance the transaction happens in a different city
    if np.random.rand() < 0.30:
        other_cities = [c for c in cities if c != home_city]
        txn_locations.append(np.random.choice(other_cities))
    else:
        txn_locations.append(home_city)

timestamps = pd.to_datetime(
    np.random.choice(pd.date_range("2023-01-01", "2024-01-01", freq="h"), n_transactions)
)

transactions = pd.DataFrame({
    "transaction_id":     [f"T{str(i).zfill(6)}" for i in range(1, n_transactions + 1)],
    "account_id":         txn_account_ids,
    "transaction_amount": np.random.randint(100, 150000, n_transactions),
    "merchant_category":  np.random.choice(merchant_categories, n_transactions),
    "transaction_type":   np.random.choice(transaction_types, n_transactions),
    "location":           txn_locations,
    "timestamp":          timestamps,
})
transactions.to_csv("data/transactions.csv", index=False)
print(f"✅ Transactions: {len(transactions)} rows")

# ── 4. FRAUD LABELS (signals baked in) ────────────────────────────────────────
fraud_probs = np.full(n_transactions, 0.05)  # base fraud rate = 5%

for i, row in transactions.iterrows():
    acc    = row["account_id"]
    cust   = acc_to_cust[acc]
    home   = cust_to_city[cust]
    hour   = row["timestamp"].hour
    merch  = row["merchant_category"]
    amount = row["transaction_amount"]
    loc    = row["location"]

    if merch == "Gambling":        fraud_probs[i] += 0.25
    if merch == "Jewelry":         fraud_probs[i] += 0.10
    if loc != home:                fraud_probs[i] += 0.15
    if amount > 50000:             fraud_probs[i] += 0.10
    if hour >= 22 or hour <= 5:    fraud_probs[i] += 0.08

fraud_probs = np.clip(fraud_probs, 0, 1)
fraud_flags = (np.random.rand(n_transactions) < fraud_probs).astype(int)

fraud_labels = pd.DataFrame({
    "transaction_id": transactions["transaction_id"],
    "fraud_flag":     fraud_flags,
})
fraud_labels.to_csv("data/fraud_labels.csv", index=False)

fraud_rate = fraud_flags.mean() * 100
print(f"✅ Fraud Labels: {len(fraud_labels)} rows | Fraud rate: {fraud_rate:.1f}%")
print("\n🎉 All 4 datasets saved to data/ folder!")