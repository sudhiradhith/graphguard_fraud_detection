"""
src/data_processing.py
Data Ingestion, Synthetic Generation, Validation, and Cleaning Module for GraphGuard.
Compliant with FR-01, FR-02, and PRD Section 7.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.DataProcessing")

REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "customer_id",
    "account_id",
    "merchant_id",
    "merchant_category",
    "amount",
    "channel",
    "device_id",
    "ip_address",
    "location",
    "is_fraud"
]

CATEGORIES = [
    "grocery", "dining", "retail", "electronics",
    "travel", "utilities", "luxury_retail", "crypto_exchange"
]

CHANNELS = ["web", "mobile_app", "pos", "atm"]
LOCATIONS = ["New York, US", "San Francisco, US", "Chicago, US", "London, UK", "Toronto, CA", "Sydney, AU", "Berlin, DE"]

def generate_synthetic_fraud_dataset(
    n_transactions: int = 12000,
    fraud_rate: float = 0.05,
    seed: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a realistic FinTech transaction and entity graph dataset
    containing documented coordinated fraud patterns:
      1. Device Sharing Syndicate (Shared device across multiple compromised accounts)
      2. IP Velocity / Bot Attack (Bursts of automated transactions from single IP)
      3. Money Mule Ring (Funneling into high-risk merchants)
      4. Account Takeover (ATO) (Sudden device/IP/amount anomaly on legitimate customer)
    
    Includes intentional edge cases for FR-01 and FR-02 data cleaning verification.
    """
    np.random.seed(seed)
    logger.info(f"Generating synthetic financial dataset with {n_transactions} transactions (target fraud rate: {fraud_rate:.1%})...")

    # 1. Base Entity Pools
    n_customers = int(n_transactions * 0.15)  # ~1,800 customers
    n_merchants = int(n_transactions * 0.04)   # ~480 merchants
    n_devices = int(n_transactions * 0.12)     # ~1,440 devices
    n_ips = int(n_transactions * 0.14)         # ~1,680 IPs

    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    account_ids = {cid: f"ACC_{i:05d}" for i, cid in enumerate(customer_ids, start=1)}
    merchant_ids = [f"MERCH_{i:04d}" for i in range(1, n_merchants + 1)]
    device_ids = [f"DEV_{i:05d}" for i in range(1, n_devices + 1)]
    ip_addresses = [f"192.168.{i // 256}.{i % 256}" for i in range(1, n_ips + 1)]

    # Assign primary device and IP to legitimate customers
    cust_primary_dev = {cid: np.random.choice(device_ids) for cid in customer_ids}
    cust_primary_ip = {cid: np.random.choice(ip_addresses) for cid in customer_ids}
    cust_primary_loc = {cid: np.random.choice(LOCATIONS) for cid in customer_ids}

    # Merchant categories
    merchant_cats = {mid: np.random.choice(CATEGORIES, p=[0.25, 0.20, 0.18, 0.12, 0.08, 0.10, 0.04, 0.03]) for mid in merchant_ids}

    # Timestamp range: 30 days starting 2026-09-01
    start_date = datetime(2026, 9, 1, 0, 0, 0)
    total_seconds = 30 * 24 * 3600

    # Determine fraud counts
    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    records = []
    
    # --- Generate Legitimate Transactions ---
    for i in range(n_legit):
        cid = np.random.choice(customer_ids)
        aid = account_ids[cid]
        mid = np.random.choice(merchant_ids)
        cat = merchant_cats[mid]

        # Diurnal timestamp distribution (less at 3-6 AM, peak afternoon/evening)
        day_offset = np.random.randint(0, 30)
        hour_probs = np.array([
            0.01, 0.01, 0.01, 0.005, 0.005, 0.01, 0.02, 0.03,
            0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.06,
            0.07, 0.08, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01
        ])
        hour_probs = hour_probs / hour_probs.sum()
        hour = int(np.random.choice(24, p=hour_probs))
        minute = int(np.random.randint(0, 60))
        second = int(np.random.randint(0, 60))
        txn_time = start_date + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute), seconds=int(second))

        # Realistic log-normal amount by category
        if cat in ["grocery", "dining", "retail"]:
            amt = np.round(np.random.lognormal(mean=3.5, sigma=0.6), 2)  # ~$30-$100
        elif cat in ["utilities"]:
            amt = np.round(np.random.lognormal(mean=4.2, sigma=0.5), 2)  # ~$60-$150
        elif cat in ["electronics", "luxury_retail", "crypto_exchange"]:
            amt = np.round(np.random.lognormal(mean=5.2, sigma=0.8), 2)  # ~$150-$500
        else:
            amt = np.round(np.random.lognormal(mean=4.5, sigma=0.7), 2)  # ~$90-$250
        amt = max(1.50, min(amt, 4500.00))

        # Device & IP consistency: 88% primary device/IP, 12% secondary
        dev = cust_primary_dev[cid] if np.random.rand() < 0.88 else np.random.choice(device_ids)
        ip = cust_primary_ip[cid] if np.random.rand() < 0.88 else np.random.choice(ip_addresses)
        loc = cust_primary_loc[cid]
        channel = np.random.choice(CHANNELS, p=[0.45, 0.35, 0.15, 0.05])

        records.append({
            "timestamp": txn_time,
            "customer_id": cid,
            "account_id": aid,
            "merchant_id": mid,
            "merchant_category": cat,
            "amount": amt,
            "channel": channel,
            "device_id": dev,
            "ip_address": ip,
            "location": loc,
            "is_fraud": 0,
            "fraud_scenario": "legitimate"
        })

    # --- Generate Coordinated Fraud Patterns ---
    fraud_records = []
    
    # 1. Device Sharing Syndicate (Shared device across 15+ accounts)
    syndicate_devices = np.random.choice(device_ids, size=3, replace=False)
    syndicate_customers = np.random.choice(customer_ids, size=40, replace=False)
    syndicate_merchants = [m for m in merchant_ids if merchant_cats[m] in ["electronics", "crypto_exchange", "luxury_retail"]][:10]
    
    n_syndicate = int(n_fraud * 0.35)
    for _ in range(n_syndicate):
        cid = np.random.choice(syndicate_customers)
        aid = account_ids[cid]
        dev = np.random.choice(syndicate_devices)
        mid = np.random.choice(syndicate_merchants)
        day_offset = int(np.random.randint(10, 25))
        hour = int(np.random.randint(0, 24))
        minute = int(np.random.randint(0, 60))
        second = int(np.random.randint(0, 60))
        txn_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
        amt = np.round(np.random.uniform(350.0, 3200.0), 2)
        
        fraud_records.append({
            "timestamp": txn_time,
            "customer_id": cid,
            "account_id": aid,
            "merchant_id": mid,
            "merchant_category": merchant_cats[mid],
            "amount": amt,
            "channel": "web",
            "device_id": dev,
            "ip_address": np.random.choice(ip_addresses),
            "location": np.random.choice(LOCATIONS),
            "is_fraud": 1,
            "fraud_scenario": "device_sharing_ring"
        })

    # 2. IP Velocity / Bot Attack (Single IP firing high-velocity transactions)
    bot_ips = np.random.choice(ip_addresses, size=3, replace=False)
    n_bot = int(n_fraud * 0.30)
    for _ in range(n_bot):
        target_ip = np.random.choice(bot_ips)
        cid = np.random.choice(customer_ids)
        aid = account_ids[cid]
        mid = np.random.choice(merchant_ids)
        # Rapid bursts within specific 2-hour window
        burst_day = 18
        burst_hour = 2
        burst_min = int(np.random.randint(0, 60))
        burst_sec = int(np.random.randint(0, 60))
        txn_time = start_date + timedelta(days=burst_day, hours=burst_hour, minutes=burst_min, seconds=burst_sec)
        amt = np.round(np.random.uniform(150.0, 1800.0), 2)

        fraud_records.append({
            "timestamp": txn_time,
            "customer_id": cid,
            "account_id": aid,
            "merchant_id": mid,
            "merchant_category": merchant_cats[mid],
            "amount": amt,
            "channel": "web",
            "device_id": np.random.choice(device_ids),
            "ip_address": target_ip,
            "location": "Moscow, RU" if np.random.rand() < 0.5 else "Unknown Proxy",
            "is_fraud": 1,
            "fraud_scenario": "ip_velocity_bot"
        })

    # 3. Money Mule / Funnel Network (Transactions funneling into shared suspicious accounts/merchants)
    mule_merchants = [m for m in merchant_ids if merchant_cats[m] in ["crypto_exchange", "luxury_retail"]][:4]
    mule_accounts = np.random.choice(customer_ids, size=25, replace=False)
    n_mule = int(n_fraud * 0.20)
    for _ in range(n_mule):
        cid = np.random.choice(mule_accounts)
        aid = account_ids[cid]
        mid = np.random.choice(mule_merchants)
        day_offset = int(np.random.randint(5, 28))
        hour = int(np.random.randint(9, 21))
        txn_time = start_date + timedelta(days=day_offset, hours=hour, minutes=int(np.random.randint(0, 60)), seconds=int(np.random.randint(0, 60)))
        amt = np.round(np.random.uniform(800.0, 4800.0), 2)

        fraud_records.append({
            "timestamp": txn_time,
            "customer_id": cid,
            "account_id": aid,
            "merchant_id": mid,
            "merchant_category": merchant_cats[mid],
            "amount": amt,
            "channel": "mobile_app",
            "device_id": cust_primary_dev[cid],
            "ip_address": cust_primary_ip[cid],
            "location": cust_primary_loc[cid],
            "is_fraud": 1,
            "fraud_scenario": "money_mule_ring"
        })

    # 4. Account Takeover (ATO) (Sudden high amount, foreign device/IP at odd hour)
    n_ato = n_fraud - (n_syndicate + n_bot + n_mule)
    ato_customers = np.random.choice(customer_ids, size=max(1, n_ato), replace=False)
    for i in range(n_ato):
        cid = ato_customers[i]
        aid = account_ids[cid]
        mid = np.random.choice([m for m in merchant_ids if merchant_cats[m] in ["electronics", "luxury_retail", "crypto_exchange"]])
        day_offset = int(np.random.randint(12, 28))
        hour = int(np.random.choice([2, 3, 4])) # Odd hour
        txn_time = start_date + timedelta(days=day_offset, hours=hour, minutes=int(np.random.randint(0, 60)), seconds=int(np.random.randint(0, 60)))
        amt = np.round(np.random.uniform(1200.0, 5000.0), 2) # Spike

        fraud_records.append({
            "timestamp": txn_time,
            "customer_id": cid,
            "account_id": aid,
            "merchant_id": mid,
            "merchant_category": merchant_cats[mid],
            "amount": amt,
            "channel": "web",
            "device_id": np.random.choice(device_ids), # Unrecognized device
            "ip_address": np.random.choice(ip_addresses), # Unrecognized IP
            "location": "Lagos, NG" if np.random.rand() < 0.5 else "Kyiv, UA",
            "is_fraud": 1,
            "fraud_scenario": "account_takeover"
        })

    all_records = records + fraud_records
    df = pd.DataFrame(all_records)

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"TXN_{i:06d}" for i in range(1, len(df) + 1)]

    # Inject realistic data quality edge-cases for FR-01/FR-02 testing
    # - 5 duplicate rows
    duplicates = df.iloc[:5].copy()
    # - 3 negative/zero amounts
    df.loc[10, "amount"] = -25.0
    df.loc[25, "amount"] = 0.0
    # - A few missing values in optional fields
    df.loc[30, "location"] = np.nan
    df.loc[45, "channel"] = np.nan

    df = pd.concat([df, duplicates], ignore_index=True)

    # Reorder columns
    col_order = ["transaction_id"] + [c for c in df.columns if c not in ["transaction_id", "fraud_scenario"]] + ["fraud_scenario"]
    df = df[col_order]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset successfully saved to {output_path} ({len(df)} records).")

    return df

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Implements FR-01: Validates required columns, datatypes, missing values,
    duplicate records, and label values. Returns a detailed diagnostic report.
    """
    report = {}
    report["total_records"] = len(df)
    report["columns"] = list(df.columns)
    
    # Check missing required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["missing_required_columns"] = missing_cols
    report["schema_valid"] = (len(missing_cols) == 0)

    # Check null counts
    null_counts = df.isnull().sum().to_dict()
    report["null_counts"] = null_counts
    report["has_nulls"] = any(v > 0 for v in null_counts.values())

    # Check duplicates
    if "transaction_id" in df.columns:
        dup_tx_ids = df["transaction_id"].duplicated().sum()
        report["duplicate_transaction_ids"] = int(dup_tx_ids)
    full_dups = df.duplicated().sum()
    report["exact_duplicate_rows"] = int(full_dups)

    # Check invalid amounts
    if "amount" in df.columns:
        invalid_amts = (df["amount"] <= 0).sum()
        report["invalid_amounts_non_positive"] = int(invalid_amts)

    # Check label distribution
    if "is_fraud" in df.columns:
        labels = df["is_fraud"].value_counts().to_dict()
        report["label_distribution"] = labels
        fraud_count = labels.get(1, 0)
        report["fraud_rate"] = float(fraud_count / len(df)) if len(df) > 0 else 0.0

    logger.info(f"Dataset validation completed. Schema valid: {report['schema_valid']}, Fraud rate: {report.get('fraud_rate', 0):.2%}")
    return report

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements FR-02: Preprocessing and cleaning:
      - Drops exact duplicate records
      - Cleans invalid amounts (amount <= 0 removed or clipped)
      - Parses timestamps into standard datetime
      - Imputes missing categorical values with 'UNKNOWN'
      - Removes duplicate transaction_ids by retaining the earliest occurrence
    """
    initial_len = len(df)
    logger.info(f"Starting data cleaning on {initial_len} records...")

    # 1. Drop exact duplicates
    df = df.drop_duplicates().copy()

    # 2. Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # 3. Clean invalid amounts
    df = df[df["amount"] > 0].copy()

    # 4. Handle duplicates in transaction_id if present
    if "transaction_id" in df.columns:
        df = df.drop_duplicates(subset=["transaction_id"], keep="first").copy()

    # 5. Impute missing categoricals
    for col in ["location", "channel", "merchant_category", "device_id", "ip_address"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # 6. Ensure labels are binary integers
    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].astype(int)

    # 7. Sort by timestamp strictly
    df = df.sort_values("timestamp").reset_index(drop=True)

    cleaned_len = len(df)
    logger.info(f"Data cleaning complete. Retained {cleaned_len}/{initial_len} valid records ({initial_len - cleaned_len} removed).")
    return df

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological data splitting ensuring zero future-to-past temporal leakage.
    Train: [0, train_ratio)
    Val:   [train_ratio, train_ratio + val_ratio)
    Test:  [train_ratio + val_ratio, 1.0]
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_idx].copy()
    val_df = df_sorted.iloc[train_idx:val_idx].copy()
    test_df = df_sorted.iloc[val_idx:].copy()

    logger.info(f"Chronological split complete:")
    logger.info(f"  Train: {len(train_df)} rows ({train_df['is_fraud'].sum()} frauds, {train_df['is_fraud'].mean():.2%}) [{train_df['timestamp'].min()} to {train_df['timestamp'].max()}]")
    logger.info(f"  Val:   {len(val_df)} rows ({val_df['is_fraud'].sum()} frauds, {val_df['is_fraud'].mean():.2%}) [{val_df['timestamp'].min()} to {val_df['timestamp'].max()}]")
    logger.info(f"  Test:  {len(test_df)} rows ({test_df['is_fraud'].sum()} frauds, {test_df['is_fraud'].mean():.2%}) [{test_df['timestamp'].min()} to {test_df['timestamp'].max()}]")

    return train_df, val_df, test_df

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "transactions.csv")
    df_raw = generate_synthetic_fraud_dataset(n_transactions=12000, fraud_rate=0.05, seed=42, output_path=raw_path)
    report = validate_dataset(df_raw)
    print("Validation Report:", json.dumps(report, indent=2, default=str))
    
    df_clean = clean_data(df_raw)
    clean_path = os.path.join("data", "processed", "transactions_cleaned.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"Saved cleaned data to {clean_path}")
