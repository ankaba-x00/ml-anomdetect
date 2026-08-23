from pydantic import BaseModel
from datetime import datetime


class PredictionRequest(BaseModel):
    model: str
    country: str
    date_from: datetime
    date_to: datetime

class PredictionResponse(BaseModel):
    country: str
    threshold: float
    detected: int
    anomalies: list[str]
    predictions: bool
    status: str

class MTPredictionResponse(BaseModel):
    country: str
    threshold: float
    detected: int
    anomalies: list[str]
    predictions: bool
    l3_intensity: list[float]
    l7_intensity: list[float]
    attack_type: list[str]
    status: str