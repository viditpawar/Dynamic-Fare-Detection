import io
import os
import tempfile
from datetime import date

import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

upload_cols = [
    "trip_distance",
    "passenger_count",
    "pulocationid",
    "dolocationid",
    "pickup_hour",
    "pickup_dow",
    "pickup_month",
    "pickup_weekofyear",
    "pickup_year",
    "total_amount",
]

BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "{color}_tripdata_{year}-{month:02d}.parquet"
)

load_dotenv()


def _default_year_month():
    today = date.today().replace(day=1)
    last_month = today - relativedelta(months=1)
    return last_month.year, last_month.month


def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT", "5432"),
        dbname=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        sslmode=os.getenv("PG_SSLMODE", "require"),
    )
    print("✅ Connected to Postgres.")
    return conn


def download_parquet(color: str, year: int, month: int) -> pd.DataFrame:
    url = BASE_URL.format(color=color, year=year, month=month)
    print(f"[{color}] Downloading {url}")

    # Stream to temp file to keep memory lower
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    tmp.write(chunk)
        tmp.flush()

        base_cols = [
            "passenger_count",
            "trip_distance",
            "total_amount",
            "PULocationID",
            "DOLocationID",
        ]
        if color == "yellow":
            dt_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        else:
            dt_cols = ["lpep_pickup_datetime", "lpep_dropoff_datetime"]

        cols = base_cols + dt_cols
        df = pd.read_parquet(tmp.name, columns=cols)

    print(f"[color={color}] Loaded {len(df):,} rows with {len(cols)} columns")
    return df


def preprocess_latest_month(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # normalize datetime columns
    if "tpep_pickup_datetime" in df.columns:
        pickup_col = "tpep_pickup_datetime"
        dropoff_col = "tpep_dropoff_datetime"
    elif "lpep_pickup_datetime" in df.columns:
        pickup_col = "lpep_pickup_datetime"
        dropoff_col = "lpep_dropoff_datetime"
    else:
        raise ValueError("No pickup/dropoff datetime columns found in df_raw")

    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")

    df = df.rename(
        columns={
            pickup_col: "pickup_datetime",
            dropoff_col: "dropoff_datetime",
        }
    )

    # normalize location columns
    rename_map = {}
    if "PULocationID" in df.columns:
        rename_map["PULocationID"] = "pulocationid"
    if "DOLocationID" in df.columns:
        rename_map["DOLocationID"] = "dolocationid"
    df = df.rename(columns=rename_map)

    # filtering
    df = df[
        (df["trip_distance"] > 0)
        & (df["total_amount"] > 0)
        & (df["passenger_count"] > 0)
        & df["pickup_datetime"].notna()
        & df["dropoff_datetime"].notna()
    ].copy()

    dt = df["pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dow"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month

    iso = dt.dt.isocalendar()
    df["pickup_weekofyear"] = iso.week.astype(int)
    df["pickup_year"] = iso.year.astype(int)

    # sample 5%
    df = df.sample(frac=0.05, random_state=42)

    missing = [c for c in upload_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after preprocessing: {missing}")

    df = df[upload_cols].reset_index(drop=True)

    df = df.astype(
        {
            "trip_distance": "int64",
            "passenger_count": "int64",
            "pulocationid": "int64",
            "dolocationid": "int64",
            "pickup_hour": "int64",
            "pickup_dow": "int64",
            "pickup_month": "int64",
            "pickup_weekofyear": "int64",
            "pickup_year": "int64",
        }
    )

    return df


def delete_existing_month(conn, year: int, month: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM taxi_trips_manhattan_ml
            WHERE pickup_year = %s
              AND pickup_month = %s;
            """,
            (year, month),
        )
    print(f"Deleted existing feature rows for {year}-{month:02d}")


def insert_trips(conn, df: pd.DataFrame):
    import numpy as np

    if df.empty:
        print("Nothing to insert.")
        return

    cols = upload_cols

    # Convert NumPy types → native Python scalars
    def to_python(v):
        if isinstance(v, np.generic):  # catches np.float64, np.int64, etc.
            return v.item()
        return v

    # Build clean Python-native tuples
    records = [
        tuple(to_python(row[c]) for c in cols)
        for _, row in df.iterrows()
    ]

    insert_sql = f"""
        INSERT INTO taxi_trips_manhattan_ml (
            {", ".join(cols)}
        )
        VALUES %s;
    """

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, records, page_size=10_000)

    print(f"Inserted {len(df):,} feature rows.")


def ingest_month(year: int, month: int):
    print(f"=== Ingesting features for {year}-{month:02d} ===")

    all_dfs = []
    for color in ("yellow", "green"):
        raw = download_parquet(color, year, month)
        proc = preprocess_latest_month(raw)
        proc = proc[
            (proc["pickup_year"] == year)
            & (proc["pickup_month"] == month)
        ].copy()
        all_dfs.append(proc)

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total feature rows (yellow+green): {len(full_df):,}")

    conn = get_db_connection()
    try:
        delete_existing_month(conn, year, month)
        insert_trips(conn, full_df)
        conn.commit()
        print("Commit OK.")
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="YYYY (e.g. 2021)")
    parser.add_argument("--month", type=int, help="1–12")
    args = parser.parse_args()

    if args.year and args.month:
        y, m = args.year, args.month
    else:
        y, m = _default_year_month()
        print(f"No year/month provided, defaulting to previous month: {y}-{m:02d}")

    ingest_month(y, m)
