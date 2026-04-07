# NYC Dynamic Taxi Fare Detection System

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-grade machine learning service that predicts NYC yellow and green taxi fares in real-time using advanced regression modeling and cloud-native architecture.

[Features](#features) • [Architecture](#cloud-architecture) • [Getting Started](#getting-started) • [API Documentation](#api-documentation) • [Deployment](#deployment)

</div>

---

## Overview

The NYC Dynamic Taxi Fare Detection System is a sophisticated ML-powered service that provides accurate fare estimates **before a ride begins**. This project showcases best practices in deploying machine learning models as scalable, observable, and performant cloud services.

Users interact with an intuitive web interface to:
- Select pickup and dropoff locations in Manhattan
- Specify date, time, and passenger count
- Receive real-time fare predictions with high accuracy

---

## Features

### 🤖 **Intelligent Predictions**
- **High Accuracy**: LightGBM regression model with ~92% R² on original fare values
- **Real-time Inference**: Sub-100ms prediction latency
- **Feature Engineering**: Computes trip distance via Mapbox Directions API
- **Data-driven**: Trained on 36 months of NYC TLC Yellow & Green taxi data

### ⚡ **Performance & Optimization**
- **Model Caching**: Models loaded once at startup, zero runtime overhead
- **Inference Cache**: PostgreSQL-backed cache prevents redundant computations
- **SQL Optimization**: Parameterized queries for security and performance
- **Containerized**: Docker-ready for seamless deployment

### 🎨 **User Experience**
- **Clean Web UI**: Responsive, mobile-friendly interface
- **Instant Feedback**: Real-time fare calculations
- **Location Search**: Searchable Manhattan taxi zone finder
- **Professional Design**: Polished, production-ready interface

### ☁️ **Cloud-Native Architecture**
- **FastAPI Framework**: Modern, async-capable Python web framework
- **PostgreSQL**: Persistent cache and inference logs
- **DigitalOcean Spaces**: Scalable model registry and artifact storage
- **Mapbox Integration**: Accurate route distance computation
- **Load Balancing**: VPC-deployed infrastructure with redundancy

### 📊 **Model Details**
| Aspect | Value |
|--------|-------|
| **Model Type** | LightGBM LGBMRegressor |
| **Target Variable** | log(total_amount) |
| **Training Data** | NYC TLC Yellow & Green taxis (36 months) |
| **Performance (R²)** | ~92% |
| **Feature Set** | trip_distance, passenger_count, pickup_hour, pickup_dow, pickup_location, dropoff_location |
| **Data Split** | Time-based: 80% train, 20% test |

---

## Cloud Architecture

```
┌─────────────────────────────────────────────┐
│         User (Web Browser)                  │
└────────────────┬────────────────────────────┘
                 │ HTTP
┌────────────────▼────────────────────────────┐
│   Load Balancer (DigitalOcean)              │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┼────────┐
        │                 │
┌───────▼──────┐  ┌──────▼────────┐
│ FastAPI App  │  │ FastAPI App   │
│ (Container)  │  │ (Container)   │
└───────┬──────┘  └──────┬────────┘
        │                 │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │   PostgreSQL    │
        │  (Inference     │
        │   Cache)        │
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        │                 │
    ┌───▼────┐      ┌─────▼──┐
    │ Mapbox │      │DO Space│
    │ Routes │      │(Models)│
    └────────┘      └────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **FastAPI Service** | REST API and web UI serving predictions |
| **LightGBM Model** | Trained regression model for fare prediction |
| **PostgreSQL** | Caches inference results and logs requests |
| **Mapbox API** | Computes accurate driving distances |
| **DigitalOcean Spaces** | Stores trained models and artifacts |
| **Docker** | Containerization for reproducible deployment |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PostgreSQL (for caching)
- Mapbox API key ([Get one here](https://account.mapbox.com/auth/signup/))
- Docker & Docker Compose (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/viditpawar/Dynamic-Fare-Detection.git
   cd Dynamic-Fare-Detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd taxi-fare-app
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the `taxi-fare-app` directory:
   ```env
   # PostgreSQL Configuration
   PG_USER=your_pg_user
   PG_PASSWORD=your_pg_password
   PG_HOST=localhost
   PG_PORT=5432
   PG_DATABASE=taxi_fare_cache
   PG_SSLMODE=disable
   
   # Mapbox API Key
   MAPBOX_API_KEY=your_mapbox_token
   ```

5. **Initialize the database**
   ```bash
   psql -U postgres -c "CREATE DATABASE taxi_fare_cache;"
   ```

### Running the Application

#### Local Development
```bash
cd taxi-fare-app
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Access the application at: `http://localhost:8000`

#### Docker Deployment
```bash
# Build the container
docker build -t taxi-fare-app:latest .

# Run with Docker Compose
docker-compose up -d
```

---

## API Documentation

### Interactive API Docs
Once running, visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### Predict Fare Endpoint

**POST** `/predict`

**Request Body:**
```json
{
  "pickup_zone": "Upper East Side South",
  "dropoff_zone": "Midtown Center",
  "pickup_hour": 14,
  "pickup_dow": 2,
  "passenger_count": 2,
  "date": "2024-04-15"
}
```

**Response:**
```json
{
  "estimated_fare": 18.50,
  "currency": "USD",
  "confidence": 0.92,
  "model_version": "1.0",
  "computed_distance_miles": 3.2,
  "message": "Fare prediction successful"
}
```

---

## Project Structure

```
Dynamic-Fare-Detection/
├── taxi-fare-app/              # Main application
│   ├── app.py                  # FastAPI application
│   ├── model_loader.py         # Model initialization
│   ├── preprocess.py           # Feature preprocessing
│   ├── train_model.py          # Model training pipeline
│   ├── taxi_ingest.py          # Data ingestion utilities
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container configuration
│   ├── templates/
│   │   └── index.html          # Web UI
│   └── static/
│       └── style.css           # Frontend styling
├── models/                      # Trained models
│   ├── lightgbm_fare_model.joblib
│   ├── xgb_fare_model.joblib
│   └── model_metadata.json
├── manhattan_zone_lookup.csv   # Location reference data
└── README.md                    # This file
```

---

## Model Training

### Training Data
- **Source**: NYC TLC Yellow & Green Taxi Trip Records
- **Duration**: 36 months
- **Records**: Millions of transactions
- **Features**: Distance, time, location, passenger count, day of week

### Training Process
```bash
cd taxi-fare-app
python train_model.py
```

### Evaluation Metrics
- **R² Score**: ~0.92 on original fare values
- **RMSE**: Minimized through quantile clipping at 99.9th percentile
- **Train/Test Split**: Time-based (80/20)

---

## Deployment

### Production Checklist
- [ ] Configure environment variables securely
- [ ] Set up PostgreSQL with secure credentials
- [ ] Obtain Mapbox API key
- [ ] Configure firewall rules (allow only necessary ports)
- [ ] Set `PG_SSLMODE=require` for external databases
- [ ] Enable health check endpoints
- [ ] Set up monitoring and logging

### Scaling Considerations
- **Horizontal Scaling**: Run multiple FastAPI containers behind a load balancer
- **Caching**: PostgreSQL cache reduces API calls to external services
- **Model Updates**: Store new models in DigitalOcean Spaces and reload gracefully

### Environment Variables Reference
| Variable | Description | Required |
|----------|-------------|----------|
| `PG_USER` | PostgreSQL username | Yes |
| `PG_PASSWORD` | PostgreSQL password | Yes |
| `PG_HOST` | PostgreSQL host address | Yes |
| `PG_PORT` | PostgreSQL port | No (default: 5432) |
| `PG_DATABASE` | Database name | Yes |
| `PG_SSLMODE` | SSL mode for DB connection | No (default: require) |
| `MAPBOX_API_KEY` | Mapbox API token | Yes |

---

## Performance Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Model Inference** | <10ms | With in-memory model |
| **Total Request** | 50-150ms | Includes API calls |
| **Cache Hit** | <5ms | Database lookup only |
| **Mapbox Distance API** | 100-200ms | Network dependent |

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core language |
| **FastAPI** | Web framework |
| **LightGBM** | ML model |
| **PostgreSQL** | Caching layer |
| **Mapbox** | Route optimization |
| **Docker** | Containerization |
| **Uvicorn** | ASGI server |
| **Jinja2** | Template engine |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Vidit Pawar**
- GitHub: [@viditpawar](https://github.com/viditpawar)
- Project: [Dynamic-Fare-Detection](https://github.com/viditpawar/Dynamic-Fare-Detection)

---

## Acknowledgments

- NYC TLC for providing publicly available taxi data
- Mapbox for route distance APIs
- LightGBM and scikit-learn communities for excellent ML libraries
- FastAPI documentation and community

---

## Support & Troubleshooting

### Common Issues

**PostgreSQL Connection Error**
- Verify PostgreSQL is running: `psql --version`
- Check connection string in `.env`
- Ensure database exists: `psql -l | grep taxi_fare_cache`

**Mapbox API Errors**
- Verify API key is valid and has routing permissions
- Check rate limits: https://account.mapbox.com/usage/

**Model Loading Fails**
- Ensure model files exist in `models/` directory
- Check file permissions: `ls -la models/`
- Verify requirements are installed

### Performance Tuning

- **Increase Cache Size**: Adjust PostgreSQL `shared_buffers`
- **Connection Pooling**: Use PgBouncer for connection management
- **Model Optimization**: Consider quantization for faster inference

---

## Roadmap

- [ ] Add support for additional NYC boroughs
- [ ] Implement surge pricing predictions
- [ ] Add historical fare trend analysis
- [ ] Multi-model ensemble for improved accuracy
- [ ] GraphQL API endpoint
- [ ] Advanced analytics dashboard
- [ ] Real-time traffic integration

---

**Last Updated**: April 2024  
**Version**: 1.0.0
