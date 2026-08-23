API_TITLE = "Anomaly Detection API"
API_VERSION = "2.0.0"
API_DESCRIPTION = """
API for running real-time anomaly detection on Cloudflare Radar data.
Choose model, country and date to infer whether anomalies occurred on that day. 
Additionally, l3/l7 intensity and attack type predictions are displayed if a 
multi-task model is selected.
"""
API_CONTACT = {
    "name": "AnKaBa",
    "email": "ankaba_x@proton.me",
}
API_LICENSE = {
    "name": "MIT License",
}
TAGS_METADATA = [
    {
        "name": "inference",
        "description": "Endpoints for running anomaly detection",
    },
    {
        "name": "system",
        "description": "Health check and system metadata",
    }
]
