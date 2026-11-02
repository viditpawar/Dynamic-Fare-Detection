import pytest

from conftest import METERS_PER_MILE
from preprocess import get_trip_distance, preprocess_input, zone_df

TRIP = dict(
    pickup_zone_name="Alphabet City",
    dropoff_zone_name="Battery Park",
    pickup_date="2025-01-12",
    pickup_time="14:30",
    passenger_count=2,
)


def test_trip_distance_converts_meters_to_miles(mapbox):
    mapbox({"routes": [{"distance": 5 * METERS_PER_MILE}]})

    assert get_trip_distance(40.7, -73.9, 40.8, -74.0) == pytest.approx(5)


@pytest.mark.parametrize("payload", [{"routes": []}, {"message": "Not Authorized"}])
def test_trip_distance_is_none_when_mapbox_has_no_route(mapbox, payload):
    mapbox(payload)

    assert get_trip_distance(40.7, -73.9, 40.8, -74.0) is None


def test_preprocess_builds_expected_features():
    row = preprocess_input(**TRIP).iloc[0]

    assert row["trip_distance"] == pytest.approx(3)
    assert row["passenger_count"] == 2
    assert row["pulocationid"] == 4
    assert row["dolocationid"] == 12
    assert row["pickup_hour"] == 14
    assert row["pickup_dow"] == 6  # 2025-01-12 is a Sunday
    assert row["pickup_month"] == 1
    assert row["pickup_weekofyear"] == 2
    assert row["pickup_year"] == 2025


@pytest.mark.parametrize("field", ["pickup_zone_name", "dropoff_zone_name"])
def test_preprocess_rejects_unknown_zone(field):
    with pytest.raises(ValueError, match="Invalid pickup/dropoff zone"):
        preprocess_input(**{**TRIP, field: "Atlantis"})


def test_preprocess_raises_when_distance_unavailable(mapbox):
    mapbox({"routes": []})

    with pytest.raises(ValueError, match="Mapbox failed"):
        preprocess_input(**TRIP)


def test_zone_lookup_has_unique_manhattan_zones():
    assert zone_df["Zone"].is_unique
    assert zone_df["LocationID"].is_unique
    assert (zone_df["Borough"] == "Manhattan").all()
