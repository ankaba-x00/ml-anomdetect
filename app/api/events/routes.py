from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from app.src.data.feature_engineering import COUNTRIES
from app.deployment.pipeline import detect_anomalies
from app.api.events.schema import PredictionRequest, PredictionResponse

router = APIRouter()

@router.get("/healthz")
def health_check():
    return {"running": True}

@router.get("/countries")
def get_countries():
    return {"countries": COUNTRIES}

@router.post("/infer", response_model=PredictionResponse)
def infer(request: PredictionRequest):
    country = request.country.upper()

    if country not in COUNTRIES:
        raise HTTPException(status_code=400, detail=f"Unknown country: {country}")

    date_from = request.date_from.replace(tzinfo=timezone.utc)
    date_to   = request.date_to.replace(tzinfo=timezone.utc)

    try:
        result = detect_anomalies(country, date_from, date_to)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictionResponse(
        country=result["country"],
        threshold=result["threshold"],
        detected=result["num_anomalies"],
        anomalies=result["intervals"],
        status=result["status"],
    )
