from pydantic import BaseModel
from datetime import datetime

class PredictionRequest(BaseModel):
    country: str
    date_from: datetime
    date_to: datetime

class PredictionResponse(BaseModel):
    country: str
    threshold: float
    detected: int
    anomalies: list[str]
    status: str
