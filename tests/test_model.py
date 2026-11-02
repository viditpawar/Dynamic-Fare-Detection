import json

import numpy as np
import pytest

from conftest import APP_DIR, METERS_PER_MILE
from model_loader import load_model
from preprocess import preprocess_input

TRIP = dict(
    pickup_zone_name="Alphabet City",
    dropoff_zone_name="Battery Park",
    pickup_date="2025-01-12",
    pickup_time="14:30",
    passenger_count=1,
)


@pytest.fixture(scope="module")
def model():
    return load_model()


def fare(model, mapbox, miles):
    mapbox({"routes": [{"distance": miles * METERS_PER_MILE}]})
    return float(np.expm1(model.predict(preprocess_input(**TRIP))[0]))


def test_preprocess_output_matches_model_features(model):
    assert list(preprocess_input(**TRIP).columns) == list(model.feature_name_)


def test_metadata_describes_served_model(model):
    metadata = json.loads((APP_DIR.parent / "models" / "model_metadata.json").read_text())

    assert metadata["feature_cols"] == list(model.feature_name_)


def test_prediction_is_a_plausible_dollar_fare(model, mapbox):
    assert 5 < fare(model, mapbox, miles=3) < 100


def test_longer_trip_costs_more(model, mapbox):
    assert fare(model, mapbox, miles=8) > fare(model, mapbox, miles=1)
