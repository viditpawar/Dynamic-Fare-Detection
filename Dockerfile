FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Copy requirements from taxi-fare-app
COPY taxi-fare-app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY taxi-fare-app/ ./taxi-fare-app/

# Copy models from project root
COPY models/ ./models/

EXPOSE 8000

WORKDIR /app/taxi-fare-app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
