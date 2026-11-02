# NYC Taxi Fare Prediction

A FastAPI web app that estimates the fare of a Manhattan yellow or green taxi trip before it starts, using a LightGBM regression model trained on NYC TLC trip records.

Originally built as a group project for MIS547.

## How it works

1. The user picks pickup and dropoff zones, a date and time, and a passenger count.
2. `preprocess.py` looks up each zone's coordinates and asks the Mapbox Directions API for the driving distance.
3. If Postgres is configured, `app.py` checks for a cached prediction with the same features.
4. Otherwise the model predicts `log(total_amount + 1)`, which is converted back to dollars and cached.

```
Browser ── form POST ──> FastAPI (taxi-fare-app/app.py)
                           ├── Mapbox Directions API (trip distance)
                           ├── LightGBM model (loaded from models/ at startup)
                           └── Postgres (optional prediction cache)
```

## Model

The served model is `models/lightgbm_fare_model.joblib`. It was trained in [fare_model_experiments.ipynb](fare_model_experiments.ipynb) on about 7 million sampled trips, using a time-based 80/20 split. Fares above the 99.9th percentile were removed first.

**Target:** `log1p(total_amount)`

**Features:** `trip_distance`, `passenger_count`, `pulocationid`, `dolocationid`, `pickup_hour`, `pickup_dow`, `pickup_month`, `pickup_weekofyear`, `pickup_year`

Results on the held-out 20%, in dollars:

| Model | R² | RMSE |
|---|---|---|
| LightGBM (served) | 0.917 | $5.99 |
| XGBoost | 0.871 | $7.77 |
| Mean-fare baseline | -0.019 | $21.01 |

LightGBM's MAE is $3.26 and its MAPE is 11.4%.

### Known limitations

- **Distance differs between training and serving.** The model was evaluated on each trip's recorded distance, but the app uses the Mapbox route between zone centroids. Live accuracy is likely lower than the table shows, and has not been measured.
- **Manhattan only.** The app accepts the 67 Manhattan zones in `manhattan_zone_lookup.csv`.
- **Same-zone trips are weak.** Both ends of the route are the same centroid, so the distance is close to zero.

## Running locally

Requires Python 3.11 and a [Mapbox access token](https://account.mapbox.com/).

```bash
git clone https://github.com/viditpawar/Dynamic-Fare-Detection.git
cd Dynamic-Fare-Detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r taxi-fare-app/requirements.txt
cp .env.example .env      # then set MAPBOX_TOKEN
cd taxi-fare-app
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000.

### With Docker

Run from the repository root:

```bash
docker build -t taxi-fare-app .
docker run --rm -p 8000:8000 --env-file .env taxi-fare-app
```

## Configuration

| Variable | Purpose | Required |
|---|---|---|
| `MAPBOX_TOKEN` | Mapbox Directions API token | Yes |
| `PG_HOST`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD` | Postgres connection | No, caching is off without them |
| `PG_PORT` | Postgres port (default `5432`) | No |
| `PG_SSLMODE` | Postgres SSL mode (default `require`) | No |

The cache expects a `fare_prediction_cache` table with one column per model feature, `predicted_fare`, `created_at`, and a unique constraint across the features. The repository does not create this table. If it is missing, the app logs the error and still returns predictions.

## Routes

| Route | Description |
|---|---|
| `GET /` | The fare form |
| `POST /predict` | Takes form fields `pickup_zone`, `dropoff_zone`, `pickup_date`, `pickup_time` and `passengers`, and returns the page with the estimate |

Both routes return HTML. There is no JSON API.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use the real model file and stub Mapbox, so they need no network access, token or database.

## Retraining

These scripts need a Postgres database containing `taxi_trips_manhattan_ml`, which the repository does not create.

- `taxi_ingest.py` downloads a month of TLC trip data, samples 5% of it and loads it into Postgres.
- `train_model.py` retrains on the trailing 36 months, logs metrics to Postgres and uploads the model to a DigitalOcean Space. It needs `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and optionally `SPACES_MODEL_BUCKET`.

The app does not download models from the Space. To serve a new model, replace `models/lightgbm_fare_model.joblib`.

## Project structure

```
├── taxi-fare-app/
│   ├── app.py                  # FastAPI routes and prediction cache
│   ├── model_loader.py         # Loads the model at startup
│   ├── preprocess.py           # Zone lookup, Mapbox call, feature building
│   ├── taxi_ingest.py          # TLC data ingestion
│   ├── train_model.py          # Retraining script
│   ├── templates/, static/     # Web UI
│   └── requirements.txt
├── models/                     # Served model and its metadata
├── tests/
├── fare_model_experiments.ipynb  # Model training and comparison
└── Dockerfile
```
