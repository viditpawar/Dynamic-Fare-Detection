import boto3
import os
import joblib

def load_model():
    session = boto3.session.Session()

    client = session.client(
        "s3",
        region_name="sfo3",
        endpoint_url="https://sfo3.digitaloceanspaces.com",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    bucket = "cloud-group-project-aditya"
    key = "models/lightgbm_fare_model_updated.joblib"

    model_path = "model.joblib"

    # ALWAYS download the latest model
    if os.path.exists(model_path):
        os.remove(model_path)

    client.download_file(bucket, key, model_path)

    print("Downloaded model size:", os.path.getsize(model_path))

    model = joblib.load(model_path)
    print("Loaded model type:", type(model))
    model = joblib.load(model_path)
    print("DEBUG → Loaded model type:", type(model))
    print("DEBUG → Model attributes:", dir(model))

    return model
