from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from model_loader import load_model
from preprocess import preprocess_input

app = FastAPI()

zone_df = pd.read_csv("manhattan_zone_lookup.csv")
ZONES = sorted(zone_df["Zone"].unique())

# Serve /static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML templates
templates = Jinja2Templates(directory="templates")

# Load model once at startup
model = load_model()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    pickup_zone: str = Form(...),
    dropoff_zone: str = Form(...),
    pickup_date: str = Form(...),
    pickup_time: str = Form(...),
    passengers: int = Form(...)
):
    try:
        # preprocess the input
        X = preprocess_input(
            pickup_zone_name=pickup_zone,
            dropoff_zone_name=dropoff_zone,
            pickup_date=pickup_date,
            pickup_time=pickup_time,
            passenger_count=passengers
        )

        # model prediction (undo log transform)
        log_pred = model.predict(X)[0]
        pred = float(np.expm1(log_pred))  # convert back from log1p()

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": round(pred, 2),
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(e),
            },
        )
