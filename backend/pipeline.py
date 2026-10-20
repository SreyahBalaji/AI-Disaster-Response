
from collections import defaultdict
from pathlib import Path
import csv
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.multi_resource import (
    model as ppo_model,
    select_ppo_actions,
    select_baseline_actions,
)
from multi_resource_cooperative_env import MultiResourceCooperativeEnv


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

router = APIRouter(
    prefix="/api/pipeline",
    tags=["Connected Three-Ward Pipeline"]
)

WARDS = (3, 5, 9)

RESOURCES = (
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases",
)

ALLOWED_REQUESTS = {
    "water",
    "food",
    "rescue",
    "shelter",
    "medicine",
}


class ReportInput(BaseModel):
    report_id: str = Field(
        min_length=1,
        max_length=80
    )

    detected_ward_no: int

    affected_families: int | None = Field(
        default=None,
        ge=0,
        le=100000
    )

    affected_patients: int | None = Field(
        default=None,
        ge=0,
        le=100000
    )

    requested_resources: list[str] = Field(
        default_factory=list
    )

    text: str = Field(
        default="",
        max_length=3000
    )


class PipelineRequest(BaseModel):
    reports: list[ReportInput] = Field(
        min_length=1,
        max_length=100
    )


def read_json(relative_path):
    path = DATA / relative_path

    try:
        with path.open(
            "r",
            encoding="utf-8-sig"
        ) as file:
            return json.load(file)

    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset unavailable: {error}"
        )


def read_csv(relative_path):
    path = DATA / relative_path

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            return list(csv.DictReader(file))

    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset unavailable: {error}"
        )


def estimate_demand(reports):
    demand = {
        ward: {
            resource: 0
            for resource in RESOURCES
        }
        for ward in WARDS
    }

    processed = []
    seen_ids = set()

    for report in reports:
        ward = report.detected_ward_no

        if ward not in WARDS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{report.report_id}: current PPO "
                    "supports Wards 3, 5 and 9 only"
                )
            )

        if report.report_id in seen_ids:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Duplicate report ID: "
                    f"{report.report_id}"
                )
            )

        seen_ids.add(report.report_id)

        requested = set(
            report.requested_resources
        )

        if (
            not requested
            or not requested.issubset(
                ALLOWED_REQUESTS
            )
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{report.report_id}: invalid "
                    "or missing resources"
                )
            )

        family_resources = {
            "water",
            "food",
            "rescue",
            "shelter",
        }

        if (
            requested.intersection(
                family_resources
            )
            and report.affected_families is None
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{report.report_id}: "
                    "affected_families required"
                )
            )

        if (
            "medicine" in requested
            and report.affected_patients is None
            and report.affected_families is None
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{report.report_id}: "
                    "medical demand cannot "
                    "be quantified"
                )
            )

        families = (
            report.affected_families or 0
        )

        # Existing project's assumptions:
        # Four people per family.
        # Three litres of water per person.
        # Three food meals per person.
        people = families * 4

        quantities = {
            "water_litres": (
                people * 3
                if "water" in requested
                else 0
            ),
            "food_meals": (
                people * 3
                if "food" in requested
                else 0
            ),
            "rescue_cases": (
                families
                if "rescue" in requested
                else 0
            ),
            "shelter_households": (
                families
                if "shelter" in requested
                else 0
            ),
            "medical_cases": (
                (
                    report.affected_patients
                    if report.affected_patients
                    is not None
                    else people
                )
                if "medicine" in requested
                else 0
            ),
        }

        for resource, quantity in quantities.items():
            demand[ward][resource] += quantity

        processed.append({
            "report_id": report.report_id,
            "ward_no": ward,
            "estimated_demand": quantities,
        })

    return demand, processed


def get_graph_context():
    priority_rows = read_csv(
        "aluva/aluva_ward_priority.csv"
    )

    embedding_rows = read_csv(
        "aluva/aluva_gat_embeddings.csv"
    )

    priorities = {
        int(row["ward_no"]): row
        for row in priority_rows
    }

    embeddings = {
        int(row["ward_no"]): row
        for row in embedding_rows
    }

    context = []

    for ward in WARDS:
        if (
            ward not in priorities
            or ward not in embeddings
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Missing graph data "
                    f"for Ward {ward}"
                )
            )

        priority = priorities[ward]
        embedding = embeddings[ward]

        context.append({
            "ward_no": ward,
            "ward_name": priority["ward_name"],
            "flood_priority_score": float(
                priority["flood_priority_score"]
            ),
            "gat_features": {
                key: float(value)
                for key, value in embedding.items()
                if key.startswith("gat_feature_")
                and value not in (None, "")
            },
        })

    return context


def simulate(algorithm, demand):
    env = MultiResourceCooperativeEnv()

    # Uploaded scenario replaces initial demand
    # in this isolated simulation.
    env.dynamic_demand_enabled = False
    env.reset()

    for ward in env.env.demand:
        for resource in RESOURCES:
            env.env.demand[ward][resource] = 0
            env.env.delivered[ward][resource] = 0

    for ward in WARDS:
        for resource in RESOURCES:
            env.env.demand[ward][resource] = (
                demand[ward][resource]
            )

    initial_stock = {
        centre["centre_id"]: {
            resource: int(
                centre["resources"][resource]
            )
            for resource in RESOURCES
        }
        for centre in env.env.centres
    }

    observations = env.get_observations()

    while True:
        if algorithm == "ppo":
            actions = select_ppo_actions(
                env,
                observations
            )
        else:
            actions = select_baseline_actions(
                env
            )

        (
            observations,
            _,
            terminated,
            truncated,
            info
        ) = env.step(actions)

        if terminated or truncated:
            break

    combined_allocations = defaultdict(int)

    for event in env.events:
        if event.get("type") != "allocation":
            continue

        key = (
            event["centre_id"],
            int(event["ward_no"]),
            event["resource"]
        )

        combined_allocations[key] += int(
            event["quantity"]
        )

    allocations = [
        {
            "centre_id": centre,
            "ward_no": ward,
            "resource": resource,
            "quantity": quantity,
        }
        for (
            centre,
            ward,
            resource
        ), quantity in sorted(
            combined_allocations.items()
        )
    ]

    # Check stock conservation.
    for centre_id, stocks in initial_stock.items():
        for resource in RESOURCES:
            sent = sum(
                allocation["quantity"]
                for allocation in allocations
                if (
                    allocation["centre_id"] == centre_id
                    and allocation["resource"] == resource
                )
            )

            if sent > stocks[resource]:
                raise HTTPException(
                    status_code=500,
                    detail="Stock conservation failed"
                )

    ward_results = []

    for ward in WARDS:
        resource_results = []

        for resource in RESOURCES:
            requested = demand[ward][resource]

            delivered = int(
                env.env.delivered[ward][resource]
            )

            if delivered > requested:
                raise HTTPException(
                    status_code=500,
                    detail="Demand conservation failed"
                )

            resource_results.append({
                "resource": resource,
                "demand": requested,
                "delivered": delivered,
                "unmet": requested - delivered,
                "satisfaction_percent": (
                    round(
                        100 * delivered / requested,
                        2
                    )
                    if requested > 0
                    else None
                ),
            })

        ward_results.append({
            "ward_no": ward,
            "resources": resource_results,
        })

    return {
        "algorithm": (
            "Masked Cooperative PPO"
            if algorithm == "ppo"
            else "Fairness-Aware Heuristic Baseline"
        ),
        "steps": env.steps,
        "fairness_gap": round(
            info["fairness_gap"], 4
        ),
        "allocations": allocations,
        "wards": ward_results,
        "remaining_stock": {
            centre["centre_id"]: {
                resource: int(
                    centre["resources"][resource]
                )
                for resource in RESOURCES
            }
            for centre in env.env.centres
        },
    }


@router.get("/status")
def pipeline_status():
    return {
        "supported_wards": list(WARDS),
        "ppo_model_loaded": ppo_model is not None,
        "max_reports_per_request": 100,
        "accepts": "Extracted disaster report data",
        "retraining_enabled": False,
    }


@router.get("/sample")
def sample_reports():
    reports = read_json(
        "crisis_reports/aluva_extracted_reports.json"
    )

    selected = [
        report
        for report in reports
        if report.get("detected_ward_no") in WARDS
    ]

    fields = {
        "report_id",
        "detected_ward_no",
        "affected_families",
        "affected_patients",
        "requested_resources",
        "text",
    }

    return {
        "reports": [
            {
                key: value
                for key, value in report.items()
                if key in fields
            }
            for report in selected
        ],
        "source": "existing_synthetic_reports",
    }


@router.post("/run")
def run_pipeline(payload: PipelineRequest):
    demand, processed = estimate_demand(
        payload.reports
    )

    graph_context = get_graph_context()

    baseline = simulate(
        "baseline",
        demand
    )

    ppo = (
        simulate("ppo", demand)
        if ppo_model is not None
        else None
    )

    return {
        "report_count": len(processed),
        "processed_reports": processed,
        "combined_demand": demand,
        "graph_context": graph_context,
        "ppo_model_loaded": ppo_model is not None,
        "ppo": ppo,
        "baseline": baseline,
        "model_integration": {
            "nlp": (
                "Validated extracted-report "
                "fields provide demand input"
            ),
            "gat": (
                "Embeddings retrieved as context; "
                "not yet included in PPO observations"
            ),
            "allocation": (
                "Three-ward PPO inference and "
                "fairness baseline"
            ),
            "federated": (
                "Not yet connected to this "
                "allocation policy"
            ),
        },
        "note": (
            "Isolated synthetic simulation with "
            "fresh initial inventory. "
            "No real dispatch or retraining."
        ),
    }
