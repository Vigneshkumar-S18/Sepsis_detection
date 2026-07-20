# Real-Time FastAPI Clinical Server Endpoint
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger
from phase10_realtime.storage.database import initialize_database
from phase10_realtime.api.patient_routes import router as patient_router

app = FastAPI(
    title="THAARU Sepsis AI Real-Time Clinical Integration Platform",
    description="Asynchronous telemetry ingestion, rolling temporal sequence building, and BiLSTM risk prediction platform.",
    version="1.0.0"
)

# Enable CORS for frontend dashboard (Phase 11 integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints router
app.include_router(patient_router, prefix="/api")


@app.get("/health")
async def health_check():
    """
    Exposes API system readiness state.
    """
    return {
        "status": "healthy",
        "service": "THAARU-Sepsis-AI-Realtime-Server",
        "timestamp": uvicorn.__version__
    }


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Real-Time Platform Services...")
    # Initialize DB tables
    initialize_database()
    logger.info("Service Initialization Complete. Ready to ingest telemetry streams.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
