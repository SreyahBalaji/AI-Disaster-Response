
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/api/federated-research",
    tags=["Federated Research Results"],
)

ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "marl"
    / "federated_multi_resource"
    / "fedavg_gat_multi_resource_results.json"
)


@router.get("/results")
def get_federated_research_results():
    """
    Read-only results of the saved Phase 27
    federated multi-resource PPO experiment.

    Does not train, approve or allocate resources.
    """
    if not RESULTS_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Federated research results not found. "
                "Run the Phase 27 training experiment first."
            ),
        )

    try:
        with RESULTS_PATH.open(
            "r", encoding="utf-8"
        ) as file:
            results = json.load(file)

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load federated research results",
        ) from exc

    required = (
        "participants",
        "round_history",
        "evaluation",
        "rounds",
        "episodes_per_centre",
    )

    if (
        not isinstance(results, dict)
        or any(key not in results for key in required)
        or not isinstance(results["round_history"], list)
        or not isinstance(results["evaluation"], dict)
    ):
        raise HTTPException(
            status_code=500,
            detail="Federated research results have an invalid format",
        )

    return {
        "status": "available",
        "source": "saved_phase27_experiment",
        "experiment": results,
        "active_allocation_policy": (
            "Coverage-first fairness heuristic"
        ),
        "federated_model_operational": False,
        "simulation_only": True,
    }