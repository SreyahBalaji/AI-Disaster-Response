
from pathlib import Path
import csv
import json

from fastapi import APIRouter, HTTPException

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"

router = APIRouter(
    prefix="/api/ai-insights",
    tags=["AI Research Insights"]
)


def read_json(path):
    file_path = BASE / path

    if not file_path.is_file():
        raise HTTPException(404, f"Missing artifact: {path}")

    try:
        with file_path.open(encoding="utf-8-sig") as file:
            return json.load(file)
    except (OSError, ValueError) as error:
        raise HTTPException(500, str(error))


def read_csv(path):
    file_path = BASE / path

    if not file_path.is_file():
        raise HTTPException(404, f"Missing artifact: {path}")

    try:
        with file_path.open(
            encoding="utf-8-sig",
            newline=""
        ) as file:
            return list(csv.DictReader(file))
    except (OSError, ValueError) as error:
        raise HTTPException(500, str(error))


@router.get("/overview")
def overview():
    return {
        "components": [
            {
                "name": "Graph Attention Network",
                "status": "Ward embeddings and connectivity",
                "endpoint": "/api/ai-insights/gat"
            },
            {
                "name": "Disaster Report Extraction",
                "status": "Rule-based synthetic reports",
                "endpoint": "/api/ai-insights/nlp"
            },
            {
                "name": "Federated Learning",
                "status": "Saved training and evaluation",
                "endpoint": "/api/ai-insights/federated"
            }
        ],
        "integration_status": (
            "Saved research outputs are displayed separately. "
            "They are not yet a unified allocation pipeline."
        )
    }


@router.get("/gat")
def gat_insights():
    rows = read_csv("aluva/aluva_gat_embeddings.csv")

    try:
        wards = [
            {
                "ward_no": int(row["ward_no"]),
                "features": {
                    key: float(value)
                    for key, value in row.items()
                    if key.startswith("gat_feature_")
                    and value not in (None, "")
                }
            }
            for row in rows
        ]
    except (KeyError, ValueError, TypeError) as error:
        raise HTTPException(500, f"Invalid GAT data: {error}")

    wards.sort(key=lambda ward: ward["ward_no"])

    return {
        "model": "Graph Attention Network",
        "artifact": "aluva_gat_embeddings.csv",
        "ward_count": len(wards),
        "feature_count": (
            len(wards[0]["features"]) if wards else 0
        ),
        "wards": wards,
        "note": (
            "Learned graph embeddings are not flood "
            "probabilities or allocation recommendations."
        )
    }


@router.get("/graph")
def graph_network():
    priority_rows = read_csv(
        "aluva/aluva_ward_priority.csv"
    )
    edge_rows = read_csv(
        "aluva/aluva_graph_edges.csv"
    )
    embedding_rows = read_csv(
        "aluva/aluva_gat_embeddings.csv"
    )
    centre_rows = read_json(
        "aluva/aluva_relief_centres.json"
    )

    if not isinstance(centre_rows, list):
        raise HTTPException(
            500, "Expected a list of relief centres"
        )

    try:
        embeddings = {
            int(row["ward_no"]): {
                key: float(value)
                for key, value in row.items()
                if key.startswith("gat_feature_")
                and value not in (None, "")
            }
            for row in embedding_rows
        }

        nodes = [
            {
                "id": int(row["ward_no"]),
                "name": row["ward_name"],
                "priority": float(
                    row["flood_priority_score"]
                ),
                "flood_exposure": float(
                    row["flood_exposure_100yr"]
                ),
                "population_score": float(
                    row["population_score"]
                ),
                "features": embeddings.get(
                    int(row["ward_no"]), {}
                )
            }
            for row in priority_rows
        ]

        edges = [
            {
                "source": int(row["source_ward"]),
                "target": int(row["target_ward"]),
                "shared_boundary_m": float(
                    row["shared_boundary_m"]
                )
            }
            for row in edge_rows
        ]

        centres = [
            {
                "id": str(row["centre_id"]),
                "name": str(row["name"]),
                "ward_no": int(row["ward_no"]),
                "resources": {
                    name: float(amount)
                    for name, amount in row.get(
                        "resources", {}
                    ).items()
                },
                "simulated": True
            }
            for row in centre_rows
        ]

    except (KeyError, ValueError, TypeError) as error:
        raise HTTPException(
            500, f"Invalid graph or centre data: {error}"
        )

    ward_ids = {node["id"] for node in nodes}

    edges = [
        edge for edge in edges
        if edge["source"] in ward_ids
        and edge["target"] in ward_ids
    ]

    for centre in centres:
        if centre["ward_no"] not in ward_ids:
            raise HTTPException(
                500,
                f"{centre['id']} references missing "
                f"Ward {centre['ward_no']}"
            )

    centre_links = [
        {
            "source": centre["id"],
            "target": centre["ward_no"],
            "relationship": "located_in_ward"
        }
        for centre in centres
    ]

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "centre_count": len(centres),
        "total_visual_nodes": len(nodes) + len(centres),
        "nodes": sorted(
            nodes, key=lambda node: node["id"]
        ),
        "edges": edges,
        "relief_centres": centres,
        "centre_links": centre_links,
        "edge_meaning": "Shared ward boundary",
        "centre_link_meaning": "Relief centre host ward",
        "note": (
            "Ward edges represent geographic adjacency, "
            "not roads. Centre links represent host wards, "
            "not verified delivery routes. Relief centres "
            "are simulated and are not additional trained "
            "GAT nodes. GAT attention coefficients are "
            "not included."
        )
    }


@router.get("/nlp")
def nlp_insights():
    reports = read_json(
        "crisis_reports/aluva_extracted_reports.json"
    )

    if not isinstance(reports, list):
        raise HTTPException(
            500, "Expected a list of reports"
        )

    resource_counts = {}
    ward_counts = {}

    for report in reports:
        ward = report.get("detected_ward_no")

        if ward is not None:
            key = str(ward)
            ward_counts[key] = ward_counts.get(key, 0) + 1

        for resource in report.get(
            "requested_resources", []
        ):
            resource_counts[resource] = (
                resource_counts.get(resource, 0) + 1
            )

    return {
        "report_count": len(reports),
        "source": "synthetic",
        "method": "rule_based_baseline",
        "resource_mentions": resource_counts,
        "ward_report_counts": ward_counts,
        "reports": reports,
        "note": (
            "Reports are synthetic, not validated "
            "against live incidents."
        )
    }


@router.get("/federated")
def federated_insights():
    evaluation = read_json(
        "federated/federated_evaluation.json"
    )
    history = read_json(
        "marl/federated/multiround_training_history.json"
    )

    if not isinstance(history, list):
        raise HTTPException(
            500, "Expected training history list"
        )

    return {
        "evaluation": evaluation,
        "training_history": history,
        "evaluation_artifact": (
            "federated/federated_evaluation.json"
        ),
        "history_artifact": (
            "marl/federated/multiround_training_history.json"
        ),
        "note": (
            "Evaluation MSE and PPO rewards are from "
            "separate experiments. The evaluated federated "
            "model is not yet integrated with multi-resource MARL."
        )
    }
