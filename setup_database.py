# setup_database.py
# Step 2: Load CSVs into SQLite and run SQL exploration queries

import sqlite3
import pandas as pd
import os

DB_PATH = "data/banking_fraud.db"
os.makedirs("data", exist_ok=True)

# ── Load CSVs ─────────────────────────────────────────────────────────────────
customers    = pd.read_csv("data/customers.csv")
accounts     = pd.read_csv("data/accounts.csv")
transactions = pd.read_csv("data/transactions.csv")
fraud_labels = pd.read_csv("data/fraud_labels.csv")

# ── Create SQLite DB and insert data ──────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
customers.to_sql("customers",    conn, if_exists="replace", index=False)
accounts.to_sql("accounts",      conn, if_exists="replace", index=False)
transactions.to_sql("transactions", conn, if_exists="replace", index=False)
fraud_labels.to_sql("fraud_labels", conn, if_exists="replace", index=False)
print(f"✅ Database created: {DB_PATH}")

# ── SQL Exploration Queries ────────────────────────────────────────────────────
queries = {
    "1. Total transactions & fraud count": """
        SELECT 
            COUNT(*) AS total_transactions,
            SUM(fraud_flag) AS fraud_count,
            ROUND(AVG(fraud_flag) * 100, 2) AS fraud_rate_pct
        FROM fraud_labels
    """,

    "2. Fraud rate by merchant category": """
        SELECT 
            t.merchant_category,
            COUNT(*) AS total_txns,
            SUM(f.fraud_flag) AS fraud_count,
            ROUND(AVG(f.fraud_flag) * 100, 2) AS fraud_rate_pct
        FROM transactions t
        JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        GROUP BY t.merchant_category
        ORDER BY fraud_rate_pct DESC
    """,

    "3. Fraud rate by city (transaction location)": """
        SELECT 
            t.location,
            COUNT(*) AS total_txns,
            SUM(f.fraud_flag) AS fraud_count,
            ROUND(AVG(f.fraud_flag) * 100, 2) AS fraud_rate_pct
        FROM transactions t
        JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        GROUP BY t.location
        ORDER BY fraud_rate_pct DESC
    """,

    "4. Average transaction amount — Fraud vs Legitimate": """
        SELECT 
            f.fraud_flag,
            ROUND(AVG(t.transaction_amount), 2) AS avg_amount,
            ROUND(MIN(t.transaction_amount), 2) AS min_amount,
            ROUND(MAX(t.transaction_amount), 2) AS max_amount
        FROM transactions t
        JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        GROUP BY f.fraud_flag
    """,

    "5. Top 10 high-value fraudulent transactions": """
        SELECT 
            t.transaction_id,
            t.transaction_amount,
            t.merchant_category,
            t.location,
            t.timestamp
        FROM transactions t
        JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        WHERE f.fraud_flag = 1
        ORDER BY t.transaction_amount DESC
        LIMIT 10
    """,

    "6. Fraud rate by transaction type": """
        SELECT 
            t.transaction_type,
            COUNT(*) AS total,
            SUM(f.fraud_flag) AS fraud_count,
            ROUND(AVG(f.fraud_flag) * 100, 2) AS fraud_rate_pct
        FROM transactions t
        JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        GROUP BY t.transaction_type
        ORDER BY fraud_rate_pct DESC
    """,

    "7. Customers with most fraud transactions": """
        SELECT 
            c.customer_id,
            c.city,
            c.occupation,
            COUNT(*) AS total_txns,
            SUM(f.fraud_flag) AS fraud_count
        FROM customers c
        JOIN accounts a ON c.customer_id = a.customer_id
        JOIN transactions t ON a.account_id = t.account_id
        JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        GROUP BY c.customer_id
        ORDER BY fraud_count DESC
        LIMIT 10
    """,

    "8. Night vs Day fraud comparison": """
        SELECT 
            CASE 
                WHEN CAST(strftime('%H', timestamp) AS INTEGER) >= 22 
                  OR CAST(strftime('%H', timestamp) AS INTEGER) <= 5 
                THEN 'Night (10PM-5AM)' 
                ELSE 'Day' 
            END AS time_period,
            COUNT(*) AS total_txns,
            SUM(f.fraud_flag) AS fraud_count,
            ROUND(AVG(f.fraud_flag) * 100, 2) AS fraud_rate_pct
        FROM transactions t
        JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        GROUP BY time_period
    """,
}

print("\n" + "="*60)
print("SQL EXPLORATION RESULTS")
print("="*60)

for title, sql in queries.items():
    print(f"\n📊 {title}")
    print("-" * 50)
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))

# Save two key results as CSVs
fraud_by_merchant = pd.read_sql_query(queries["2. Fraud rate by merchant category"], conn)
fraud_by_merchant.to_csv("data/sql_fraud_by_merchant.csv", index=False)

city_fraud = pd.read_sql_query(queries["3. Fraud rate by city (transaction location)"], conn)
city_fraud.to_csv("data/sql_city_fraud.csv", index=False)

conn.close()
print("\n✅ SQL exploration complete! DB saved to data/banking_fraud.db")