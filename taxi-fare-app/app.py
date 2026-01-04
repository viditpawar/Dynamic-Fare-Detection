import os
import numpy as np
import pandas as pd
import psycopg2
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from model_loader import load_model
from preprocess import preprocess_input, zone_df  # zone_df already loaded there
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()

# Read PG_* vars from .env / environment
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE")
PG_SSLMODE = os.getenv("PG_SSLMODE", "require")

# Build a DSN-style string mostly for logging / sanity
if PG_HOST and PG_DATABASE and PG_USER:
    DATABASE_URL = (
        f"postgresql://{PG_USER}:{PG_PASSWORD}@"
        f"{PG_HOST}:{PG_PORT}/{PG_DATABASE}?sslmode={PG_SSLMODE}"
    )
else:
    DATABASE_URL = None

print("DATABASE_URL in app.py:", repr(DATABASE_URL))

# -----------------------------------------------------------------------------
# FastAPI + globals
# -----------------------------------------------------------------------------
app = FastAPI()

# Serve /static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML templates
templates = Jinja2Templates(directory="templates")

# Load model once at startup
model = load_model()

# Zones (driven by CSV via preprocess.py)
ZONES = sorted(zone_df["Zone"].unique())


# -----------------------------------------------------------------------------
# DB / cache helpers
# -----------------------------------------------------------------------------
def get_db_connection():
    """
    Create a psycopg2 connection using the PG_* vars.
    DATABASE_URL is only used as a flag to check config exists.
    """
    if not DATABASE_URL:
        raise RuntimeError("Postgres env vars not set correctly")

    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        sslmode=PG_SSLMODE,
    )


def _normalize_feature_row(features_df):
    """
    Take the 1-row dataframe from preprocess_input and convert to
    a dict of *plain* Python types (float/int) so psycopg2 can adapt them.
    """
    r = features_df.iloc[0]

    return {
        "trip_distance": float(r["trip_distance"]),        # plain float
        "passenger_count": int(r["passenger_count"]),
        "pulocationid": int(r["pulocationid"]),
        "dolocationid": int(r["dolocationid"]),
        "pickup_hour": int(r["pickup_hour"]),
        "pickup_dow": int(r["pickup_dow"]),
        "pickup_month": int(r["pickup_month"]),
        "pickup_weekofyear": int(r["pickup_weekofyear"]),
        "pickup_year": int(r["pickup_year"]),
    }


def get_cached_prediction(features_df):
    """Return cached fare (float) if present, else None."""
    if not DATABASE_URL:
        # Caching disabled if PG_* not set
        return None

    row = _normalize_feature_row(features_df)

    query = """
        SELECT predicted_fare
        FROM fare_prediction_cache
        WHERE trip_distance     = %s
          AND passenger_count   = %s
          AND pulocationid      = %s
          AND dolocationid      = %s
          AND pickup_hour       = %s
          AND pickup_dow        = %s
          AND pickup_month      = %s
          AND pickup_weekofyear = %s
          AND pickup_year       = %s
        LIMIT 1;
    """
    params = (
        row["trip_distance"],
        row["passenger_count"],
        row["pulocationid"],
        row["dolocationid"],
        row["pickup_hour"],
        row["pickup_dow"],
        row["pickup_month"],
        row["pickup_weekofyear"],
        row["pickup_year"],
    )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)  # <-- parameterized, no np.x in SQL
                result = cur.fetchone()
                if result:
                    return float(result[0])
    except Exception as e:
        print("Cache lookup error:", e)

    return None


def save_cached_prediction(features_df, fare):
    """Upsert the prediction into the cache."""
    if not DATABASE_URL:
        # Caching disabled if PG_* not set
        return

    row = _normalize_feature_row(features_df)

    query = """
        INSERT INTO fare_prediction_cache (
            trip_distance,
            passenger_count,
            pulocationid,
            dolocationid,
            pickup_hour,
            pickup_dow,
            pickup_month,
            pickup_weekofyear,
            pickup_year,
            predicted_fare
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (trip_distance,
                     passenger_count,
                     pulocationid,
                     dolocationid,
                     pickup_hour,
                     pickup_dow,
                     pickup_month,
                     pickup_weekofyear,
                     pickup_year)
        DO UPDATE SET
            predicted_fare = EXCLUDED.predicted_fare,
            created_at     = NOW();
    """
    params = (
        row["trip_distance"],
        row["passenger_count"],
        row["pulocationid"],
        row["dolocationid"],
        row["pickup_hour"],
        row["pickup_dow"],
        row["pickup_month"],
        row["pickup_weekofyear"],
        row["pickup_year"],
        float(fare),
    )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)  # <-- also parameterized
                conn.commit()
    except Exception as e:
        print("Cache save error:", e)
# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "zones": ZONES,  # pass zones into template
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    pickup_zone: str = Form(...),
    dropoff_zone: str = Form(...),
    pickup_date: str = Form(...),
    pickup_time: str = Form(...),
    passengers: int = Form(...)
):
    try:
        # 1. Preprocess the input -> 1-row feature DF
        X = preprocess_input(
            pickup_zone_name=pickup_zone,
            dropoff_zone_name=dropoff_zone,
            pickup_date=pickup_date,
            pickup_time=pickup_time,
            passenger_count=passengers,
        )

        # 2. Try cache first
        cached = get_cached_prediction(X)
        if cached is not None:
            pred = cached
        else:
            # 3. Model prediction (undo log1p)
            log_pred = model.predict(X)[0]
            pred = float(np.expm1(log_pred))

            # 4. Save to cache
            save_cached_prediction(X, pred)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "zones": ZONES,
                "prediction": round(pred, 2),
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "zones": ZONES,
                "error": str(e),
            },
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
