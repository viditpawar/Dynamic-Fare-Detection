import os
import joblib

def load_model():
    base_dir = os.path.dirname(__file__)  # this will be /app inside Docker
    model_path = os.path.join(base_dir, "models", "lightgbm_fare_model_updated.joblib")

    print("Loading model from:", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    print("Loaded model type:", type(model))
    return model
