import os
import joblib

def load_model():
    # models/ sits next to taxi-fare-app/, both locally and in the Docker image
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "lightgbm_fare_model.joblib")

    print("Loading model from:", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    print("Loaded model type:", type(model))
    return model
