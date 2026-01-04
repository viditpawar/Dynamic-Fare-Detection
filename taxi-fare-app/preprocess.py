import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Load centroid lookup table (must be inside project folder)
ZONE_FILE = "manhattan_zone_lookup.csv"
zone_df = pd.read_csv(ZONE_FILE)

# Create dictionaries for fast lookup
zone_to_id = dict(zip(zone_df["Zone"], zone_df["LocationID"]))
id_to_lat = dict(zip(zone_df["LocationID"], zone_df["lat"]))
id_to_lon = dict(zip(zone_df["LocationID"], zone_df["lon"]))

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")


def get_trip_distance(pu_lat, pu_lon, do_lat, do_lon):
    """
    Calls Mapbox Directions API to get driving distance (in miles).
    """

    url = (
        f"https://api.mapbox.com/directions/v5/mapbox/driving/"
        f"{pu_lon},{pu_lat};{do_lon},{do_lat}"
        f"?geometries=geojson&access_token={MAPBOX_TOKEN}"
    )

    response = requests.get(url)
    data = response.json()

    try:
        meters = data["routes"][0]["distance"]
        miles = meters / 1609.34
        return miles
    except Exception as e:
        print("Mapbox error:", e)
        return None


def preprocess_input(
    pickup_zone_name: str,
    dropoff_zone_name: str,
    pickup_date: str,      # NEW: e.g. "2025-01-12"
    pickup_time: str,      # NEW: e.g. "14:30"
    passenger_count: int
):
    """
    Converts UI inputs into a model-ready feature vector:
    - Converts zone name → LocationID → lat/lon
    - Calls Mapbox to compute driving distance
    - Extracts datetime features
    """

    # --- Zone → LocationID ---
    pu_id = zone_to_id.get(pickup_zone_name)
    do_id = zone_to_id.get(dropoff_zone_name)

    if pu_id is None or do_id is None:
        raise ValueError("Invalid pickup/dropoff zone")

    # --- LocationID → coordinates ---
    pu_lat, pu_lon = id_to_lat[pu_id], id_to_lon[pu_id]
    do_lat, do_lon = id_to_lat[do_id], id_to_lon[do_id]

    # --- Build full datetime ---
    # pickup_date = "2025-01-12"
    # pickup_time = "14:30"
    pickup_datetime_str = f"{pickup_date} {pickup_time}"
    dt = pd.to_datetime(pickup_datetime_str)

    # --- Get trip distance from Mapbox ---
    trip_distance = get_trip_distance(pu_lat, pu_lon, do_lat, do_lon)
    if trip_distance is None:
        raise ValueError("Mapbox failed to compute distance")

    # --- Extract datetime features ---
    pickup_hour = dt.hour
    pickup_dow = dt.weekday()
    pickup_month = dt.month
    pickup_weekofyear = dt.isocalendar().week
    pickup_year = dt.year

    # --- Build feature dataframe ---
    features = pd.DataFrame([{
        "trip_distance": trip_distance,
        "passenger_count": passenger_count,
        "pulocationid": pu_id,
        "dolocationid": do_id,
        "pickup_hour": pickup_hour,
        "pickup_dow": pickup_dow,
        "pickup_month": pickup_month,
        "pickup_weekofyear": pickup_weekofyear,
        "pickup_year": pickup_year,
    }])

    return features