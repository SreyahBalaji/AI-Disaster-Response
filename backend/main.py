
from pathlib import Path
from backend.rolling_allocation import router as rolling_router
import sys

import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.federated_results import router as federated_results_router

BASE_DIR = Path(__file__).resolve().parent.parent

MARL_DIR = BASE_DIR / "scripts" / "marl"

if str(MARL_DIR) not in sys.path:
    sys.path.insert(0, str(MARL_DIR))


from cooperative_ppo_network import CooperativePPO

from backend.allocation import register_allocation_routes
from backend.wards import router as wards_router
from backend.gis import router as gis_router
from backend.gis_priority import router as gis_priority_router
from backend.multi_resource import router as multi_resource_router
from backend.ai_insights import router as ai_insights_router
from backend.decision_support import router as decision_support_router
from backend.pipeline import router as pipeline_router
from backend.allocation_safety import router as safety_router
from backend.dataset_upload import router as dataset_router


MODEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "marl"
    / "selected_cooperative_ppo.pth"
)


app = FastAPI(
    title="Responda Disaster Intelligence API",
    description=(
        "GIS, GAT, NLP, cooperative resource "
        "allocation and federated learning"
    ),
    version="1.6.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
model_error = None

try:
    model = CooperativePPO()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    print("Existing cooperative PPO loaded.")

except Exception as error:
    model = None
    model_error = str(error)

    print(
        "Cooperative PPO failed:",
        model_error,
    )


app.include_router(
    register_allocation_routes(model)
)

app.include_router(wards_router)
app.include_router(gis_router)
app.include_router(gis_priority_router)
app.include_router(multi_resource_router)
app.include_router(ai_insights_router)
app.include_router(decision_support_router)
app.include_router(pipeline_router)
app.include_router(safety_router)
app.include_router(rolling_router)
app.include_router(federated_results_router)
# New dataset-upload endpoints
app.include_router(dataset_router)


@app.get("/")
def home():
    return {
        "message": "Responda API is running",
        "status": "success",
        "version": "1.6.0",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": (
            "healthy"
            if model is not None
            else "degraded"
        ),
        "service": "Responda Backend",
        "ai_model": (
            "loaded"
            if model is not None
            else "not loaded"
        ),
    }


@app.get("/api/model/status")
def model_status():
    return {
        "model_name": (
            "Cooperative PPO (water-only)"
        ),
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
        "device": "cpu",
        "error": model_error,
    }
