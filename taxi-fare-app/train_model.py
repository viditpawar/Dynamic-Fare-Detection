import os
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import boto3

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from taxi_ingest import get_db_connection  # reuse your existing DB connector


FEATURE_COLS = [
    "trip_distance",
    "passenger_count",
    "pulocationid",
    "dolocationid",
    "pickup_hour",
    "pickup_dow",
    "pickup_month",
    "pickup_weekofyear",
    "pickup_year",
]

TARGET_COL = "total_amount"


def load_data_from_db():
    """
    Load all feature rows from Postgres into a DataFrame.
    We order approximately by time so our 80/20 split is time-based.
    """
    conn = get_db_connection()
    try:
        query = """
    SELECT
        trip_distance,
        passenger_count,
        pulocationid,
        dolocationid,
        pickup_hour,
        pickup_dow,
        pickup_month,
        pickup_weekofyear,
        pickup_year,
        total_amount
    FROM taxi_trips_manhattan_ml
    WHERE make_date(pickup_year, pickup_month, 1) >=
          (date_trunc('month', CURRENT_DATE) - INTERVAL '36 months')
    ORDER BY pickup_year, pickup_month, pickup_weekofyear, pickup_hour;
"""
        df = pd.read_sql(query, conn)
        print(f"Loaded {len(df):,} rows from taxi_trip_manhattan_ml")
        return df
    finally:
        conn.close()


def train_model(df: pd.DataFrame):
    """
    Train LightGBM on all available data with an 80/20 time-based split.
    Returns (model, metrics_dict).
    """

    # Basic cleaning
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    p999 = df[TARGET_COL].quantile(0.999)
    print(f"Clipping total_amount at 99.9th percentile: {p999:.2f}")
    df[TARGET_COL] = df[TARGET_COL].clip(upper=p999)
    
    # Log-transform target
    df["log_total_amount"] = np.log1p(df[TARGET_COL])

    X = df[FEATURE_COLS]
    y_log = df["log_total_amount"]
    y_orig = df[TARGET_COL]

    # Time-based split: first 80% train, last 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_log, y_test_log = y_log.iloc[:split_idx], y_log.iloc[split_idx:]
    y_train_orig, y_test_orig = y_orig.iloc[:split_idx], y_orig.iloc[split_idx:]

    print(f"Train rows: {len(X_train):,}, Test rows: {len(X_test):,}")

    # LightGBM model (same config as your old script)
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    lgbm_model.fit(X_train, y_train_log)

    # Predictions in log-space
    y_pred_log = lgbm_model.predict(X_test)

    # Metrics in log-space
    rmse_log = float(np.sqrt(mean_squared_error(y_test_log, y_pred_log)))
    r2_log = float(r2_score(y_test_log, y_pred_log))

    # Back-transform to original-space
    y_pred_orig = np.expm1(y_pred_log)

    rmse_orig = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))
    r2_orig = float(r2_score(y_test_orig, y_pred_orig))
    mae_orig = float(mean_absolute_error(y_test_orig, y_pred_orig))

    print("=== LightGBM (log target) ===")
    print("Log-space  RMSE / R² :", rmse_log, r2_log)
    print("Original   RMSE / R² :", rmse_orig, r2_orig)
    print("Original   MAE       :", mae_orig)

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "rmse_orig": rmse_orig,
        "r2_orig": r2_orig,
        "mae_orig": mae_orig,
    }

    return lgbm_model, metrics


def save_model_locally(model) -> str:
    """
    Save the trained model to /app/models and return the local path.
    """
    os.makedirs("/app/models", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"lightgbm_fare_model_{timestamp}.joblib"
    local_path = os.path.join("/app/models", filename)

    joblib.dump(model, local_path)
    print(f"Saved model to {local_path}")
    return local_path


def upload_to_spaces(local_path: str) -> tuple[str, str]:
    """
    Upload the model file to DigitalOcean Spaces via S3-compatible API.
    Returns (bucket_name, object_key).
    """
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Use your bucket name (or env var)
    bucket_name = os.getenv("SPACES_MODEL_BUCKET", "cloud-group-project-aditya")

    # keep a dated version AND a 'latest' pointer
    base_name = os.path.basename(local_path)
    object_key_versioned = f"models/{base_name}"
    object_key_latest = "models/lightgbm_fare_model_latest.joblib"

    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    # Upload dated version
    client.upload_file(
        Filename=local_path,
        Bucket=bucket_name,
        Key=object_key_versioned,
        ExtraArgs={"ACL": "public-read"},
    )
    print(f"Uploaded model to s3://{bucket_name}/{object_key_versioned}")

    # Also upload/update a "latest" pointer
    client.upload_file(
        Filename=local_path,
        Bucket=bucket_name,
        Key=object_key_latest,
        ExtraArgs={"ACL": "public-read"},
    )
    print(f"Updated latest model at s3://{bucket_name}/{object_key_latest}")

    # Return the “latest” key for logging
    return bucket_name, object_key_latest


def log_metrics_to_db(metrics: dict, bucket_name: str, object_key: str):
    """
    Create a metrics table (if needed) and insert one row for this training run.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS taxi_model_metrics (
                    id SERIAL PRIMARY KEY,
                    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    train_rows INTEGER,
                    test_rows INTEGER,
                    rmse_log DOUBLE PRECISION,
                    r2_log DOUBLE PRECISION,
                    rmse_orig DOUBLE PRECISION,
                    r2_orig DOUBLE PRECISION,
                    mae_orig DOUBLE PRECISION,
                    bucket_name TEXT,
                    object_key TEXT
                );
                """
            )

            cur.execute(
                """
                INSERT INTO taxi_model_metrics (
                    train_rows,
                    test_rows,
                    rmse_log,
                    r2_log,
                    rmse_orig,
                    r2_orig,
                    mae_orig,
                    bucket_name,
                    object_key
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    metrics["train_rows"],
                    metrics["test_rows"],
                    metrics["rmse_log"],
                    metrics["r2_log"],
                    metrics["rmse_orig"],
                    metrics["r2_orig"],
                    metrics["mae_orig"],
                    bucket_name,
                    object_key,
                ),
            )

        conn.commit()
        print("Logged metrics to taxi_model_metrics")
    finally:
        conn.close()


def main():
    print("=== Loading data from Postgres ===")
    df = load_data_from_db()

    if df.empty:
        print("No data found in taxi_trip_manhattan_ml. Aborting training.")
        return

    print("=== Training model ===")
    model, metrics = train_model(df)

    print("=== Saving model locally ===")
    local_path = save_model_locally(model)

    print("=== Uploading model to Spaces ===")
    bucket, key = upload_to_spaces(local_path)

    print("=== Logging metrics to Postgres ===")
    log_metrics_to_db(metrics, bucket, key)

    print("Training + upload + logging complete.")


if __name__ == "__main__":
    main()
