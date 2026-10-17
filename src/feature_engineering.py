"""
src/feature_engineering.py
Leakage-Safe Feature Engineering Pipeline for GraphGuard.
Implements FR-03: Amount, temporal, velocity, customer-history, and entity-reuse features.
Strictly fits encoders and scalers on training split to prevent target/temporal leakage.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGuard.FeatureEngineering")

def compute_leakage_safe_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-aware rolling frequency and historical entity reuse features
    using past transactions only (strictly causal, zero look-ahead leakage).
    Assumes df is sorted chronologically by timestamp.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].values
    
    # 1. Temporal basic features
    dt = pd.to_datetime(df["timestamp"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_night"] = ((dt.dt.hour >= 23) | (dt.dt.hour <= 5)).astype(int)
    df["log_amount"] = np.log1p(df["amount"].values)

    # Convert timestamps to nanoseconds/seconds for fast window comparisons
    ts_seconds = dt.astype("int64") // 10**9

    n = len(df)
    cust_1h_counts = np.zeros(n, dtype=np.int32)
    cust_24h_counts = np.zeros(n, dtype=np.int32)
    dev_24h_counts = np.zeros(n, dtype=np.int32)
    ip_24h_counts = np.zeros(n, dtype=np.int32)
    
    # Dynamic customer historical amount tracking
    cust_mean_amt_diff = np.zeros(n, dtype=np.float32)
    cust_amt_ratio = np.ones(n, dtype=np.float32)

    # Entity reuse tracking (distinct customers per device/IP)
    dev_distinct_cust = np.ones(n, dtype=np.int32)
    ip_distinct_cust = np.ones(n, dtype=np.int32)

    # In-memory history trackers
    cust_history = {}    # cid -> list of (timestamp_sec, amount)
    dev_history = {}     # dev -> list of (timestamp_sec, customer_id)
    ip_history = {}      # ip  -> list of (timestamp_sec, customer_id)

    cids = df["customer_id"].values
    devs = df["device_id"].values
    ips = df["ip_address"].values
    amts = df["amount"].values

    one_hour = 3600
    one_day = 86400

    for i in range(n):
        t = ts_seconds.iloc[i]
        c = cids[i]
        d = devs[i]
        ip = ips[i]
        amt = amts[i]

        # Customer velocity & history
        if c in cust_history:
            past_c = cust_history[c]
            # 1h count
            c1h = sum(1 for (pt, _) in past_c if (t - pt) <= one_hour)
            # 24h count
            c24h = sum(1 for (pt, _) in past_c if (t - pt) <= one_day)
            cust_1h_counts[i] = c1h
            cust_24h_counts[i] = c24h
            
            # Historical spending
            past_amts = [pa for (_, pa) in past_c]
            mean_amt = np.mean(past_amts)
            cust_mean_amt_diff[i] = amt - mean_amt
            cust_amt_ratio[i] = amt / (mean_amt + 1e-5)
            past_c.append((t, amt))
        else:
            cust_history[c] = [(t, amt)]
            cust_mean_amt_diff[i] = 0.0
            cust_amt_ratio[i] = 1.0

        # Device reuse & velocity
        if d in dev_history:
            past_d = dev_history[d]
            dev_24h_counts[i] = sum(1 for (pt, _) in past_d if (t - pt) <= one_day)
            # distinct customers using this device up to now
            distinct_c = len(set(pc for (_, pc) in past_d))
            dev_distinct_cust[i] = distinct_c
            past_d.append((t, c))
        else:
            dev_history[d] = [(t, c)]
            dev_distinct_cust[i] = 1

        # IP reuse & velocity
        if ip in ip_history:
            past_ip = ip_history[ip]
            ip_24h_counts[i] = sum(1 for (pt, _) in past_ip if (t - pt) <= one_day)
            # distinct customers using this IP up to now
            distinct_c_ip = len(set(pc for (_, pc) in past_ip))
            ip_distinct_cust[i] = distinct_c_ip
            past_ip.append((t, c))
        else:
            ip_history[ip] = [(t, c)]
            ip_distinct_cust[i] = 1

    df["cust_tx_count_1h"] = cust_1h_counts
    df["cust_tx_count_24h"] = cust_24h_counts
    df["dev_tx_count_24h"] = dev_24h_counts
    df["ip_tx_count_24h"] = ip_24h_counts
    df["cust_mean_amt_diff"] = cust_mean_amt_diff
    df["cust_amt_ratio"] = cust_amt_ratio
    df["dev_distinct_cust"] = dev_distinct_cust
    df["ip_distinct_cust"] = ip_distinct_cust

    return df

class FeaturePipeline:
    """
    Fits categorical encoders and standard scalers strictly on training data,
    then transforms train, validation, and test sets.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.numeric_features = [
            "amount", "log_amount", "hour", "day_of_week", "is_weekend", "is_night",
            "cust_tx_count_1h", "cust_tx_count_24h", "dev_tx_count_24h", "ip_tx_count_24h",
            "cust_mean_amt_diff", "cust_amt_ratio", "dev_distinct_cust", "ip_distinct_cust"
        ]
        self.categorical_features = ["merchant_category", "channel"]
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        logger.info(f"Fitting FeaturePipeline on {len(train_df)} training samples...")
        
        # Fit scaler on numeric features
        X_num = train_df[self.numeric_features].values
        self.scaler.fit(X_num)

        # Fit encoder on categorical features
        X_cat = train_df[self.categorical_features].astype(str)
        self.encoder.fit(X_cat)

        cat_names = self.encoder.get_feature_names_out(self.categorical_features).tolist()
        self.feature_names = self.numeric_features + cat_names
        self.is_fitted = True
        logger.info(f"Pipeline fitted successfully with {len(self.feature_names)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform!")
        
        X_num = df[self.numeric_features].values
        X_num_scaled = self.scaler.transform(X_num)

        X_cat = df[self.categorical_features].astype(str)
        X_cat_encoded = self.encoder.transform(X_cat)

        X_full = np.hstack([X_num_scaled, X_cat_encoded])
        return X_full

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "encoder": self.encoder,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }, filepath)
        logger.info(f"FeaturePipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FeaturePipeline":
        data = joblib.load(filepath)
        pipe = cls()
        pipe.scaler = data["scaler"]
        pipe.encoder = data["encoder"]
        pipe.numeric_features = data["numeric_features"]
        pipe.categorical_features = data["categorical_features"]
        pipe.feature_names = data["feature_names"]
        pipe.is_fitted = data["is_fitted"]
        logger.info(f"FeaturePipeline loaded from {filepath}")
        return pipe

def prepare_feature_data(
    input_csv: str = "data/processed/transactions_cleaned.csv",
    save_pipeline_path: str = "models/feature_pipeline.joblib",
    output_processed_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Executes the end-to-end feature pipeline:
      1. Loads cleaned transactions.
      2. Computes chronological rolling features.
      3. Splits data into train (70%), val (15%), test (15%).
      4. Fits pipeline on train and transforms train, val, test.
      5. Saves pipeline and prepared datasets.
    """
    df = pd.read_csv(input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    logger.info("Computing leakage-safe rolling features...")
    df_feat = compute_leakage_safe_rolling_features(df)
    
    # Save enriched dataframe
    enriched_path = os.path.join(output_processed_dir, "transactions_engineered.csv")
    df_feat.to_csv(enriched_path, index=False)
    logger.info(f"Saved engineered features to {enriched_path}")

    # Chronological splits
    n = len(df_feat)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    train_df = df_feat.iloc[:train_idx].copy()
    val_df = df_feat.iloc[train_idx:val_idx].copy()
    test_df = df_feat.iloc[val_idx:].copy()

    # Fit pipeline
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)
    pipeline.save(save_pipeline_path)

    X_train = pipeline.transform(train_df)
    y_train = train_df["is_fraud"].values

    X_val = pipeline.transform(val_df)
    y_val = val_df["is_fraud"].values

    X_test = pipeline.transform(test_df)
    y_test = test_df["is_fraud"].values

    logger.info(f"Feature processing complete. Feature dimensionality: {X_train.shape[1]}")
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    return {
        "df_engineered": df_feat,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": pipeline.feature_names
    }

if __name__ == "__main__":
    result = prepare_feature_data()
    print("Feature Pipeline Finished Successfully!")
    print(f"Features ({len(result['feature_names'])}): {result['feature_names']}")
