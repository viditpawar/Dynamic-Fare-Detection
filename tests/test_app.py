import re

import pytest
from fastapi.testclient import TestClient

import app as app_module
from preprocess import preprocess_input

FORM = {
    "pickup_zone": "Alphabet City",
    "dropoff_zone": "Battery Park",
    "pickup_date": "2025-01-12",
    "pickup_time": "14:30",
    "passengers": 2,
}


@pytest.fixture
def client():
    return TestClient(app_module.app)


def displayed_fare(html):
    return float(re.search(r'fare-amount">\$([\d.]+)<', html).group(1))


def test_home_lists_zones(client):
    response = client.get("/")

    assert response.status_code == 200
    assert '<option value="Alphabet City">' in response.text
    assert "fare-amount" not in response.text


def test_predict_renders_fare_and_keeps_inputs(client):
    response = client.post("/predict", data=FORM)

    assert response.status_code == 200
    assert 5 < displayed_fare(response.text) < 100
    assert 'value="Alphabet City"' in response.text
    assert 'value="2025-01-12"' in response.text


def test_cache_hit_skips_the_model(client, monkeypatch):
    class ModelMustNotRun:
        def predict(self, X):
            raise AssertionError("model called despite cache hit")

    monkeypatch.setattr(app_module, "get_cached_prediction", lambda X: 12.34)
    monkeypatch.setattr(app_module, "model", ModelMustNotRun())

    response = client.post("/predict", data=FORM)

    assert displayed_fare(response.text) == 12.34


def test_cache_miss_saves_the_prediction(client, monkeypatch):
    saved = []
    monkeypatch.setattr(app_module, "get_cached_prediction", lambda X: None)
    monkeypatch.setattr(
        app_module, "save_cached_prediction", lambda X, fare: saved.append(fare)
    )

    response = client.post("/predict", data=FORM)

    assert displayed_fare(response.text) == pytest.approx(saved[0], abs=0.01)


def test_unknown_zone_shows_error(client):
    response = client.post("/predict", data={**FORM, "pickup_zone": "Atlantis"})

    assert response.status_code == 200
    assert "Invalid pickup/dropoff zone" in response.text
    assert "fare-amount" not in response.text


def test_mapbox_failure_shows_error(client, mapbox):
    mapbox({"message": "Not Authorized"})

    response = client.post("/predict", data=FORM)

    assert "Mapbox failed to compute distance" in response.text


def test_missing_field_is_rejected(client):
    form = {k: v for k, v in FORM.items() if k != "pickup_date"}

    assert client.post("/predict", data=form).status_code == 422


def test_feature_row_uses_plain_python_types():
    row = app_module._normalize_feature_row(
        preprocess_input("Alphabet City", "Battery Park", "2025-01-12", "14:30", 2)
    )

    # psycopg2 cannot adapt NumPy scalars, so these must be builtins.
    assert type(row["trip_distance"]) is float
    assert all(type(v) is int for k, v in row.items() if k != "trip_distance")
