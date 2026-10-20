
from collections import deque
from pathlib import Path
import csv
import json
import sys

from fastapi import APIRouter, HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MARL_DIR = BASE_DIR / "scripts" / "marl"

if str(MARL_DIR) not in sys.path:
    sys.path.insert(0, str(MARL_DIR))

from multi_resource_cooperative_env import MultiResourceCooperativeEnv
from backend.multi_resource import (
    model as ppo_model,
    select_ppo_actions,
    select_baseline_actions,
)

router = APIRouter(
    prefix="/api/decision-support",
    tags=["End-to-End Decision Support"]
)

RESOURCES = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases",
]

RESOURCE_LABELS = {
    "water_litres": "Drinking water",
    "food_meals": "Food",
    "rescue_cases": "Rescue",
    "shelter_households": "Shelter",
    "medical_cases": "Medical assistance",
}

SUPPORTED_PPO_WARDS = {3, 5, 9}


def load_json(relative_path):
    path = DATA_DIR / relative_path

    if not path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Required data missing: {relative_path}"
        )

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot read {relative_path}: {error}"
        )


def load_csv(relative_path):
    path = DATA_DIR / relative_path

    if not path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Required data missing: {relative_path}"
        )

    try:
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            return list(csv.DictReader(file))
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot read {relative_path}: {error}"
        )


def report_data():
    reports = load_json(
        "crisis_reports/aluva_extracted_reports.json"
    )

    events = load_json(
        "crisis_reports/aluva_resource_demand.json"
    )

    if not isinstance(reports, list) or not isinstance(events, list):
        raise HTTPException(
            500, "Invalid report or demand data"
        )

    event_lookup = {
        event["report_id"]: event
        for event in events
    }

    return [
        {
            **report,
            "demand_event": event_lookup.get(
                report["report_id"]
            )
        }
        for report in reports
    ]


def build_graph():
    rows = load_csv("aluva/aluva_graph_edges.csv")

    adjacency = {
        ward: set()
        for ward in range(1, 27)
    }

    for row in rows:
        a = int(row["source_ward"])
        b = int(row["target_ward"])

        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency


def graph_hops(adjacency, start, destination):
    if start == destination:
        return 0

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        ward, distance = queue.popleft()

        for neighbor in adjacency.get(ward, set()):
            if neighbor == destination:
                return distance + 1

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return None


def ward_metadata(ward_no):
    rows = load_csv(
        "aluva/aluva_ward_priority.csv"
    )

    matching = next(
        (
            row for row in rows
            if int(row["ward_no"]) == ward_no
        ),
        None
    )

    if matching is None:
        raise HTTPException(
            500, f"Ward {ward_no} not found"
        )

    return {
        "ward_no": ward_no,
        "ward_name": matching["ward_name"],
        "flood_priority_score": float(
            matching["flood_priority_score"]
        ),
        "flood_exposure": float(
            matching["flood_exposure_100yr"]
        ),
        "population_score": float(
            matching["population_score"]
        )
    }


def ward_embeddings(ward_no):
    rows = load_csv(
        "aluva/aluva_gat_embeddings.csv"
    )

    matching = next(
        (
            row for row in rows
            if int(row["ward_no"]) == ward_no
        ),
        None
    )

    if matching is None:
        return {}

    return {
        key: float(value)
        for key, value in matching.items()
        if key.startswith("gat_feature_")
        and value not in (None, "")
    }


def allocate_graph_heuristic(ward_no, demand):
    centres = load_json(
        "aluva/aluva_relief_centres.json"
    )

    adjacency = build_graph()

    # Copy available stock so the original JSON is untouched.
    inventory = {
        centre["centre_id"]: {
            resource: int(
                centre["resources"].get(resource, 0)
            )
            for resource in RESOURCES
        }
        for centre in centres
    }

    centre_info = {
        centre["centre_id"]: centre
        for centre in centres
    }

    remaining = {
        resource: int(demand.get(resource, 0))
        for resource in RESOURCES
    }

    allocations = []
    unreachable_resources = []

    for resource in RESOURCES:
        if remaining[resource] <= 0:
            continue

        # Prefer fewer adjacency hops; use stock as
        # a deterministic tie-breaker. These are not
        # geographic distances or real travel times.
        candidates = []

        for centre in centres:
            centre_id = centre["centre_id"]
            hops = graph_hops(
                adjacency,
                int(centre["ward_no"]),
                ward_no
            )

            if hops is None:
                continue

            stock = inventory[centre_id][resource]

            if stock > 0:
                candidates.append(
                    (hops, -stock, centre_id)
                )

        candidates.sort()

        if not candidates:
            unreachable_resources.append(resource)

        for hops, _, centre_id in candidates:
            if remaining[resource] <= 0:
                break

            available = inventory[centre_id][resource]
            allocated = min(
                available,
                remaining[resource]
            )

            if allocated <= 0:
                continue

            inventory[centre_id][resource] -= allocated
            remaining[resource] -= allocated

            allocations.append({
                "centre_id": centre_id,
                "centre_name": centre_info[
                    centre_id
                ]["name"],
                "centre_ward": int(
                    centre_info[centre_id]["ward_no"]
                ),
                "target_ward": ward_no,
                "resource": resource,
                "resource_label": RESOURCE_LABELS[resource],
                "quantity": allocated,
                "adjacency_hops": hops,
                "reason": (
                    f"{centre_id} has available "
                    f"{RESOURCE_LABELS[resource].lower()} "
                    f"and is {hops} ward-adjacency "
                    f"hop(s) from Ward {ward_no}. "
                    "This is not a verified road route."
                )
            })

    resource_summary = []

    for resource in RESOURCES:
        requested = int(demand.get(resource, 0))
        unmet = remaining[resource]

        resource_summary.append({
            "resource": resource,
            "label": RESOURCE_LABELS[resource],
            "requested": requested,
            "allocated": requested - unmet,
            "unmet": unmet,
            "satisfaction_percent": (
                round(
                    100 * (requested - unmet) / requested,
                    2
                )
                if requested > 0 else None
            )
        })

    return {
        "algorithm": (
            "Graph-Informed Stock-Constrained Heuristic"
        ),
        "allocations": allocations,
        "resource_summary": resource_summary,
        "remaining_stock": inventory,
        "unmet_resources": [
            item for item in resource_summary
            if item["unmet"] > 0
        ],
        "explanation": (
            "Resources are assigned from simulated "
            "relief centres according to ward-adjacency "
            "hop count and available inventory. "
            "The method obeys stock and demand limits. "
            "It is a deterministic heuristic, not "
            "a learned GAT or MARL policy."
        ),
        "warnings": [
            (
                "Ward adjacency is not a road network. "
                "Actual travel time and road availability "
                "have not been evaluated."
            ),
            (
                "Each report is evaluated independently "
                "with a fresh copy of initial stock."
            )
        ]
    }


def evaluate_supported_ppo(ward_no, demand):
    if ward_no not in SUPPORTED_PPO_WARDS:
        return {
            "available": False,
            "reason": (
                "The trained 16-action PPO policy "
                "supports Wards 3, 5 and 9 only."
            )
        }

    if ppo_model is None:
        return {
            "available": False,
            "reason": "Saved PPO model is not loaded."
        }

    outcomes = {}

    for algorithm in ("ppo", "baseline"):
        env = MultiResourceCooperativeEnv()
        env.dynamic_demand_enabled = False
        env.reset()

        # Create a controlled single-report scenario.
        # No changes to the trained model or stored files.
        for ward in env.env.demand:
            for resource in RESOURCES:
                env.env.demand[ward][resource] = 0
                env.env.delivered[ward][resource] = 0

        for resource in RESOURCES:
            env.env.demand[ward_no][resource] = int(
                demand.get(resource, 0)
            )

        observations = env.get_observations()
        successful_actions = 0

        while True:
            if algorithm == "ppo":
                actions = select_ppo_actions(
                    env, observations
                )
            else:
                actions = select_baseline_actions(env)

            observations, _, terminated, truncated, info = (
                env.step(actions)
            )

            successful_actions += info[
                "successful_actions"
            ]

            if terminated or truncated:
                break

        ward_results = []

        for resource in RESOURCES:
            requested = int(demand.get(resource, 0))
            allocated = int(
                env.env.delivered[ward_no][resource]
            )

            ward_results.append({
                "resource": resource,
                "requested": requested,
                "allocated": allocated,
                "unmet": requested - allocated
            })

        outcomes[algorithm] = {
            "algorithm": (
                "Masked Cooperative PPO"
                if algorithm == "ppo"
                else "Fairness-Aware Heuristic Baseline"
            ),
            "steps": env.steps,
            "successful_actions": successful_actions,
            "resources": ward_results,
            "total_allocated_units": sum(
                row["allocated"]
                for row in ward_results
            )
        }

    return {
        "available": True,
        "scenario": "isolated_single_report",
        "ward_no": ward_no,
        "ppo": outcomes["ppo"],
        "baseline": outcomes["baseline"],
        "note": (
            "The existing trained PPO was evaluated "
            "without retraining on an isolated report "
            "scenario. This differs from its original "
            "multi-ward training distribution. "
            "Different resource units must not be "
            "interpreted as directly comparable."
        )
    }


@router.get("/reports")
def list_reports():
    reports = report_data()

    return {
        "count": len(reports),
        "reports": [
            {
                "report_id": report["report_id"],
                "text": report["text"],
                "ward_no": report["detected_ward_no"],
                "requested_resources": report[
                    "requested_resources"
                ],
                "ppo_supported": (
                    report["detected_ward_no"]
                    in SUPPORTED_PPO_WARDS
                )
            }
            for report in reports
        ]
    }


@router.get("/recommend/{report_id}")
def recommend(report_id: str):
    report = next(
        (
            item for item in report_data()
            if item["report_id"].upper()
            == report_id.upper()
        ),
        None
    )

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    event = report["demand_event"]

    if event is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "No quantified demand event exists "
                "for this report."
            )
        )

    ward_no = report["detected_ward_no"]

    if ward_no is None:
        raise HTTPException(
            status_code=422,
            detail="Report has no identified ward"
        )

    demand = {
        resource: int(event["demand"].get(resource, 0))
        for resource in RESOURCES
    }

    if any(amount < 0 for amount in demand.values()):
        raise HTTPException(
            status_code=500,
            detail="Invalid negative demand"
        )

    priority = ward_metadata(ward_no)
    embeddings = ward_embeddings(ward_no)

    plan = allocate_graph_heuristic(
        ward_no, demand
    )

    try:
        ppo_comparison = evaluate_supported_ppo(
            ward_no, demand
        )
    except Exception as error:
        ppo_comparison = {
            "available": False,
            "reason": f"PPO evaluation failed: {error}"
        }

    explanation = {
        "report_reason": (
            f"Report {report['report_id']} identifies "
            f"Ward {ward_no} and requests "
            + ", ".join(
                report["requested_resources"]
            )
            + "."
        ),
        "demand_reason": (
            "Quantities come from the project's "
            "existing synthetic demand-estimation "
            "pipeline, using affected families "
            "or patients."
        ),
        "ward_reason": (
            f"{priority['ward_name']} has a "
            f"calculated flood-priority score of "
            f"{priority['flood_priority_score']:.4f}. "
            "This is contextual information, not "
            "a validated emergency risk rating."
        ),
        "allocation_reason": plan["explanation"],
        "gat_note": (
            "Saved GAT embeddings are displayed as "
            "ward context. They do not currently "
            "change this allocation decision."
        )
    }

    return {
        "report": {
            "report_id": report["report_id"],
            "text": report["text"],
            "source_type": report.get(
                "source_type", "synthetic"
            ),
            "extraction_method": report.get(
                "extraction_method",
                "rule_based_baseline"
            ),
            "ward_no": ward_no,
            "affected_families": report.get(
                "affected_families"
            ),
            "affected_patients": report.get(
                "affected_patients"
            ),
            "requested_resources": report[
                "requested_resources"
            ]
        },
        "ward": priority,
        "gat_features": embeddings,
        "demand": demand,
        "unquantified_resources": event.get(
            "unquantified_resources", []
        ),
        "recommendation": plan,
        "ppo_comparison": ppo_comparison,
        "explanation": explanation,
        "disclaimer": (
            "Demonstration using synthetic reports "
            "and simulated relief centres. "
            "No real emergency dispatch occurs."
        )
    }
