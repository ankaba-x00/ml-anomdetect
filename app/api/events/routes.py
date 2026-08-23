from fastapi import APIRouter, HTTPException
from datetime import timezone

from app.src.data.feature_engineering import COUNTRIES
from app.deployment.use_model import use_model
from app.api.events.schema import PredictionRequest, PredictionResponse, MTPredictionResponse


router = APIRouter()

@router.get("/healthz")
def health_check():
    return {"running": True}

@router.get("/countries")
def get_countries():
    return {"countries": COUNTRIES}

@router.post("/infer")
def infer(request: PredictionRequest):
    model = request.model.lower()
    if model not in ["ae", "vae", "mtae"]:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    country = request.country.upper()
    if country not in COUNTRIES:
        raise HTTPException(status_code=400, detail=f"Unknown country: {country}")

    date_from = request.date_from.replace(tzinfo=timezone.utc)
    date_to   = request.date_to.replace(tzinfo=timezone.utc)

    try:
        result = use_model(model, country, date_from, date_to)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    if model in ["ae", "vae"]:
        return PredictionResponse(
            country=result["country"],
            threshold=result["threshold"],
            detected=result["num_anomalies"],
            anomalies=result["intervals"],
            predictions=False,
            status=result["status"],
        )
    else:
        return MTPredictionResponse(
            country=result["country"],
            threshold=result["threshold"],
            detected=result["num_anomalies"],
            anomalies=result["intervals"],
            predictions=True,
            l3_intensity=result["l3_intensity"],
            l7_intensity=result["l7_intensity"],
            attack_type=result["attack_type"],
            status=result["status"],
        )     
