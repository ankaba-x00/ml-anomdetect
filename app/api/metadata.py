API_TITLE = "Anomaly Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API for running real-time anomaly detection on Cloudflare Radar data.
Choose a country and a date to check whether anomalies occurred.
"""

API_CONTACT = {
    "name": "AnKaBa",
    "email": "ankaba_x@proton.me",
}

API_LICENSE = {
    "name": "Custom Proprietary License",
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
