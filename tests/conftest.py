import os
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "taxi-fare-app"

# The app resolves its CSV, static and template folders from the working
# directory and imports its modules as top-level names.
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))

# Set before the app is imported so a developer's real .env is never used:
# load_dotenv() does not override variables that already exist.
os.environ.update(PG_HOST="", PG_DATABASE="", PG_USER="", MAPBOX_TOKEN="test-token")

import preprocess  # noqa: E402

METERS_PER_MILE = 1609.34


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


@pytest.fixture(autouse=True)
def mapbox(monkeypatch):
    """Stub the Mapbox call. Returns a setter for the payload it responds with."""

    def respond_with(payload):
        monkeypatch.setattr(
            preprocess.requests, "get", lambda url: FakeResponse(payload)
        )

    respond_with({"routes": [{"distance": 3 * METERS_PER_MILE}]})
    return respond_with
