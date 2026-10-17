
from pathlib import Path
import sys

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MARL_DIR = BASE_DIR / "scripts" / "marl"

sys.path.insert(0, str(MARL_DIR))

from cooperative_ppo_network import CooperativePPO


MODEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "marl"
    / "selected_cooperative_ppo.pth"
)


app = FastAPI(
    title="AI Disaster Response API",
    description=(
        "AI-driven disaster response "
        "and resource allocation system"
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Load the trained PPO model once.
model = None
model_error = None

try:
    model = CooperativePPO()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    print("SUCCESS! Federated PPO model loaded.")

except Exception as error:
    model = None
    model_error = str(error)
    print("Model loading failed:", model_error)
from backend.allocation import register_allocation_routes
app.include_router(register_allocation_routes(model))
from backend.wards import router as wards_router
app.include_router(wards_router)
from backend.gis import router as gis_router
app.include_router(gis_router)
from backend.gis_priority import router as gis_priority_router
app.include_router(gis_priority_router)
@app.get("/")
def home():
    return {
        "message": "AI Disaster Response API is running",
        "status": "success"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "service": "Disaster Response Backend",
        "ai_model": (
            "loaded" if model is not None
            else "not loaded"
        )
    }


@app.get("/api/model/status")
def model_status():
    return {
        "model_name": "Federated Cooperative PPO",
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
        "device": "cpu",
        "error": model_error
    }
