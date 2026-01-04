# NYC Taxi Fare Estimator 🚕

A production-style cloud-native machine learning service that predicts NYC yellow and green taxi fares **before the ride begins**.

This project demonstrates how a traditional ML model can be deployed as a **scalable, observable, and cache-optimized web service**.

---

## 🔍 What this app does

Users select:
- Pickup & dropoff taxi zones in Manhattan
- Date & time of travel
- Passenger count

The system:
- Computes realistic driving distance using Mapbox Directions API
- Builds a feature vector matching the training pipeline
- Runs a LightGBM regression model
- Returns a real-time fare estimate via a web UI

---

## 🧠 Machine Learning
- Model: **LightGBM regression**
- Trained on NYC TLC Yellow & Green taxi data (36 months)
- Target: `log(total_amount)`
- R² ≈ **92%** on original fare values

---

## ☁️ Cloud Architecture
- **FastAPI** inference service
- **Dockerized** deployment
- **PostgreSQL-backed inference cache**
- **DigitalOcean Spaces** model registry
- **Mapbox Directions API** for route distance
- Load-balanced droplets inside a **VPC**

---

## ⚡ Performance Optimizations
- Model loaded once at startup
- Feature-based inference cache to avoid recomputation
- Parameterized SQL for safety and performance

---

## 🖥️ Web Interface
- Clean, searchable UI
- Instant feedback
- Mobile-friendly design
- Product-style fare display

---

## 🚀 Run locally

```bash
uvicorn app:app --reload
