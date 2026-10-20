
"""
RESPONDA — Phase 23
Generate synthetic disaster forecasting scenarios.

Uses:
- Existing 26-ward Aluva graph
- Existing report-derived resource demand patterns
- Spatially connected simulated disaster spread

Produces separate training, validation and
test scenarios.

Does NOT overwrite existing datasets or models.
"""

import argparse
import json
import random
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    ROOT / "data" / "processed" / "aluva"
    / "aluva_gat_data.pt"
)

REPORTS_PATH = (
    ROOT / "data" / "processed"
    / "crisis_reports"
    / "aluva_resource_demand.json"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "aluva"
    / "aluva_forecast_scenarios_v2.pt"
)

RESOURCE_NAMES = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases",
]


def load_sources():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(GRAPH_PATH)

    if not REPORTS_PATH.exists():
        raise FileNotFoundError(REPORTS_PATH)

    # Trusted, locally generated PyG graph.
    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    with open(
        REPORTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        reports = json.load(file)

    features = graph.x.float()
    edges = graph.edge_index.long()
    wards = graph.ward_numbers.long()

    if features.shape != (26, 8):
        raise ValueError(
            f"Expected (26, 8), got {features.shape}"
        )

    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(
            "Invalid graph edge_index"
        )

    if wards.numel() != 26:
        raise ValueError(
            "Expected 26 ward numbers"
        )

    if not torch.isfinite(features).all():
        raise ValueError(
            "Non-finite static graph features"
        )

    templates = []

    for report in reports:
        resource_data = report["demand"]

        values = [
            float(resource_data[name])
            for name in RESOURCE_NAMES
        ]

        if any(value < 0 for value in values):
            raise ValueError(
                "Negative resource demand found"
            )

        if any(value > 0 for value in values):
            templates.append(values)

    if not templates:
        raise ValueError(
            "No positive demand templates found"
        )

    return features, edges, wards, templates


def build_neighbors(edges, count):
    neighbors = {
        index: set()
        for index in range(count)
    }

    for source, destination in edges.T.tolist():
        if not (
            0 <= source < count
            and 0 <= destination < count
        ):
            raise ValueError(
                "Graph edge outside ward range"
            )

        if source != destination:
            neighbors[source].add(destination)
            neighbors[destination].add(source)

    return neighbors


def generate_one_scenario(
    rng,
    steps,
    neighbors,
    templates,
    ward_count,
):
    """
    Create a synthetic, spatially spreading disaster.

    At each step:
    - Existing affected wards may spread to neighbors
    - New reports add resource demand
    - All demand remains cumulative

    These are simulations, not real incident reports.
    """

    snapshots = torch.zeros(
        steps + 1,
        ward_count,
        len(RESOURCE_NAMES),
        dtype=torch.float32,
    )

    initial_ward = rng.randrange(ward_count)

    active_wards = {initial_ward}

    intensity = rng.uniform(0.6, 1.8)
    spread_probability = rng.uniform(
        0.08, 0.28
    )

    for step in range(1, steps + 1):
        snapshots[step] = snapshots[
            step - 1
        ].clone()

        new_wards = set()

        for ward in list(active_wards):
            for neighbor in neighbors[ward]:
                if neighbor not in active_wards:
                    if rng.random() < spread_probability:
                        new_wards.add(neighbor)

        active_wards.update(new_wards)

        # Occasional isolated incident.
        if rng.random() < 0.06:
            active_wards.add(
                rng.randrange(ward_count)
            )

        # Keep some attention on newly affected wards.
        candidates = list(active_wards)

        if new_wards:
            candidates.extend(
                list(new_wards) * 2
            )

        report_count = rng.randint(1, 3)

        for _ in range(report_count):
            ward = rng.choice(candidates)

            template = rng.choice(templates)

            scale = (
                intensity
                * rng.uniform(0.5, 1.5)
            )

            values = [
                max(
                    1,
                    round(value * scale),
                ) if value > 0 else 0
                for value in template
            ]

            snapshots[step, ward] += (
                torch.tensor(
                    values,
                    dtype=torch.float32,
                )
            )

        # Small changes in severity across time.
        intensity *= rng.uniform(
            0.96, 1.04
        )

    return snapshots


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenarios",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.scenarios < 10:
        parser.error(
            "At least 10 scenarios required"
        )

    if args.steps < 2:
        parser.error(
            "At least 2 steps required"
        )

    torch.manual_seed(args.seed)

    (
        static_features,
        edges,
        wards,
        templates,
    ) = load_sources()

    neighbors = build_neighbors(
        edges,
        len(wards),
    )

    generated = []

    for scenario_id in range(
        args.scenarios
    ):
        rng = random.Random(
            args.seed + 10007 * scenario_id
        )

        snapshots = generate_one_scenario(
            rng=rng,
            steps=args.steps,
            neighbors=neighbors,
            templates=templates,
            ward_count=len(wards),
        )

        generated.append(snapshots)

    demand = torch.stack(generated)

    # Dimensions:
    # [scenarios, snapshots, wards, resources]

    scenario_order = list(
        range(args.scenarios)
    )

    random.Random(
        args.seed + 999
    ).shuffle(scenario_order)

    train_count = int(
        args.scenarios * 0.70
    )

    validation_count = int(
        args.scenarios * 0.15
    )

    train_ids = scenario_order[
        :train_count
    ]

    validation_ids = scenario_order[
        train_count:
        train_count + validation_count
    ]

    test_ids = scenario_order[
        train_count + validation_count:
    ]

    # Critical: fit normalization only on
    # training scenarios, never on test scenarios.

    train_demand = demand[train_ids]

    mean = train_demand.mean(
        dim=(0, 1, 2)
    )

    std = train_demand.std(
        dim=(0, 1, 2),
        unbiased=False,
    )

    std = torch.where(
        std < 1e-8,
        torch.ones_like(std),
        std,
    )

    normalized_demand = (
        demand - mean
    ) / std

    expanded_static = (
        static_features
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(
            args.scenarios,
            args.steps + 1,
            -1,
            -1,
        )
    )

    features = torch.cat(
        [
            expanded_static,
            normalized_demand,
        ],
        dim=-1,
    )

    inputs = features[:, :-1]
    targets = features[
        :, 1:, :, 8:13
    ]

    assert inputs.shape == (
        args.scenarios,
        args.steps,
        26,
        13,
    )

    assert targets.shape == (
        args.scenarios,
        args.steps,
        26,
        5,
    )

    assert torch.isfinite(inputs).all()
    assert torch.isfinite(targets).all()

    # Verify chronological cumulative demand.
    increases = (
        demand[:, 1:]
        - demand[:, :-1]
    )

    assert (increases >= 0).all()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "inputs": inputs,
            "targets": targets,
            "edge_index": edges,
            "ward_numbers": wards,
            "train_scenarios": train_ids,
            "validation_scenarios": validation_ids,
            "test_scenarios": test_ids,
            "demand_mean": mean,
            "demand_std": std,
            "resource_names": RESOURCE_NAMES,
            "scenario_count": args.scenarios,
            "steps": args.steps,
            "seed": args.seed,
            "synthetic": True,
        },
        OUTPUT_PATH,
    )

    print("=" * 62)
    print(
        "PHASE 23 — SYNTHETIC FORECAST "
        "DATASET GENERATION"
    )
    print("=" * 62)

    print(
        "Scenarios:",
        args.scenarios,
    )

    print(
        "Snapshots per scenario:",
        args.steps + 1,
    )

    print(
        "Input tensor:",
        tuple(inputs.shape),
    )

    print(
        "Target tensor:",
        tuple(targets.shape),
    )

    print(
        "Graph edges:",
        edges.shape[1],
    )

    print()
    print("SCENARIO-LEVEL SPLIT")
    print("-" * 62)

    print(
        "Training:",
        len(train_ids),
    )

    print(
        "Validation:",
        len(validation_ids),
    )

    print(
        "Testing:",
        len(test_ids),
    )

    print()
    print(
        "Demand means:",
        [round(x, 3) for x in mean.tolist()],
    )

    print(
        "Demand stds:",
        [round(x, 3) for x in std.tolist()],
    )

    print()
    print(
        "PASS: No negative cumulative demand"
    )

    print(
        "PASS: Train/validation/test scenarios "
        "are separate"
    )

    print(
        "PASS: Normalization fitted only "
        "on training scenarios"
    )

    print()
    print("Saved:", OUTPUT_PATH)

    print()
    print(
        "IMPORTANT: This dataset is synthetic "
        "and does not establish real-world "
        "forecast accuracy."
    )


if __name__ == "__main__":
    main()
