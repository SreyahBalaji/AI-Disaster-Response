
"""
RESPONDA — Phase 24
Persistent simulated disaster resource allocation.

Features:
- Incremental disaster reports
- Duplicate-report protection
- Persistent simulated inventory
- Fairness-based rolling allocation proposals
- Explicit approval before reserving resources
- Version checks to prevent stale approvals
- No real-world dispatch

Local single-process prototype only.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from backend.pipeline import (
    PipelineRequest,
    RESOURCES,
    WARDS,
    estimate_demand,
)
from multi_resource_cooperative_env import (
    MultiResourceCooperativeEnv,
)
from backend.approval_security import require_approval_key

router = APIRouter(
    prefix="/api/rolling",
    tags=["Rolling Disaster Allocation"],
)

ROOT = Path(__file__).resolve().parent.parent

STATE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "rolling"
    / "aluva_rolling_state.json"
)

LOCK = threading.RLock()

BATCH_SIZES = {
    "water_litres": 10,
    "food_meals": 5,
    "rescue_cases": 1,
    "shelter_households": 1,
    "medical_cases": 1,
}

VULNERABILITY = {
    3: 0.6,
    5: 0.7,
    9: 0.9,
}


class ApprovalRequest(BaseModel):
    review_id: str = Field(min_length=1)
    expected_version: int = Field(ge=0)


class ResetRequest(BaseModel):
    confirm: bool = False


def now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def empty_ward_matrix():
    return {
        str(ward): {
            resource: 0
            for resource in RESOURCES
        }
        for ward in WARDS
    }


def create_initial_state():
    env = MultiResourceCooperativeEnv()
    env.dynamic_demand_enabled = False
    env.reset()

    stock = {
        centre["centre_id"]: {
            resource: int(
                centre["resources"][resource]
            )
            for resource in RESOURCES
        }
        for centre in env.env.centres
    }

    return {
        "version": 0,
        "created_at": now(),
        "simulation_only": True,
        "supported_wards": list(WARDS),
        "initial_stock": stock,
        "remaining_stock": {
            centre: resources.copy()
            for centre, resources in stock.items()
        },
        "demand": empty_ward_matrix(),
        "reserved": empty_ward_matrix(),
        "report_ids": [],
        "report_history": [],
        "approval_history": [],
        "pending_review": None,
    }


def save_state(state):
    STATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Atomic replacement prevents a partially
    # written JSON file after interruption.
    temporary = STATE_PATH.with_name(
        STATE_PATH.name + "." + uuid.uuid4().hex + ".tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                state,
                file,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary,
            STATE_PATH,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def validate_state(state):
    for ward in WARDS:
        key = str(ward)

        for resource in RESOURCES:
            demand = int(
                state["demand"][key][resource]
            )

            reserved = int(
                state["reserved"][key][resource]
            )

            if not 0 <= reserved <= demand:
                raise HTTPException(
                    status_code=500,
                    detail="Invalid ward reservation state",
                )

    for centre, resource_stock in (
        state["remaining_stock"].items()
    ):
        for resource in RESOURCES:
            remaining = int(
                resource_stock[resource]
            )

            initial = int(
                state["initial_stock"][centre][resource]
            )

            if not 0 <= remaining <= initial:
                raise HTTPException(
                    status_code=500,
                    detail="Invalid inventory state",
                )


def load_state():
    if not STATE_PATH.exists():
        state = create_initial_state()
        save_state(state)
        return state

    try:
        with STATE_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(file)

        validate_state(state)
        return state

    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Rolling state could not be read: {error}",
        )


def unmet_demand(state, ward, resource):
    key = str(ward)

    return (
        state["demand"][key][resource]
        - state["reserved"][key][resource]
    )


def build_proposal(state):
    """
    Allocate simulated available stock to
    unmet ward-resource demand.

    The algorithm prioritizes the lowest
    satisfaction ratio first.

    Vulnerability breaks ties.
    """

    stock = {
        centre: resources.copy()
        for centre, resources in (
            state["remaining_stock"].items()
        )
    }

    planned = empty_ward_matrix()
    combined = {}

    for resource in RESOURCES:
        while True:
            eligible_wards = []

            for ward in WARDS:
                key = str(ward)

                requested = (
                    state["demand"][key][resource]
                )

                already_reserved = (
                    state["reserved"][key][resource]
                )

                proposed = planned[key][resource]

                remaining_need = (
                    requested
                    - already_reserved
                    - proposed
                )

                if requested <= 0 or remaining_need <= 0:
                    continue

                coverage = (
                    (already_reserved + proposed)
                    / requested
                )

                eligible_wards.append((
                    coverage,
                    -VULNERABILITY[ward],
                    ward,
                ))

            available_centres = [
                centre
                for centre in stock
                if stock[centre][resource] > 0
            ]

            if not eligible_wards or not available_centres:
                break

            # Worst satisfaction first.
            eligible_wards.sort()

            ward = eligible_wards[0][2]
            key = str(ward)

            # Use the centre with the most stock
            # for this resource, breaking ties
            # consistently by centre ID.
            available_centres.sort(
                key=lambda centre: (
                    -stock[centre][resource],
                    centre,
                )
            )

            centre = available_centres[0]

            remaining_need = (
                state["demand"][key][resource]
                - state["reserved"][key][resource]
                - planned[key][resource]
            )

            quantity = min(
                BATCH_SIZES[resource],
                remaining_need,
                stock[centre][resource],
            )

            if quantity <= 0:
                break

            planned[key][resource] += quantity
            stock[centre][resource] -= quantity

            allocation_key = (
                centre,
                ward,
                resource,
            )

            combined[allocation_key] = (
                combined.get(allocation_key, 0)
                + quantity
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
            resource,
        ), quantity in sorted(
            combined.items()
        )
    ]

    coverage_values = []
    ward_results = []

    for ward in WARDS:
        key = str(ward)

        for resource in RESOURCES:
            requested = (
                state["demand"][key][resource]
            )

            reserved = (
                state["reserved"][key][resource]
            )

            proposed = planned[key][resource]

            total_planned = reserved + proposed
            unmet = requested - total_planned

            if requested > 0:
                percentage = (
                    100 * total_planned / requested
                )

                coverage_values.append(percentage)

                ward_results.append({
                    "ward_no": ward,
                    "resource": resource,
                    "requested": requested,
                    "already_reserved": reserved,
                    "newly_proposed": proposed,
                    "total_if_approved": total_planned,
                    "unmet_if_approved": unmet,
                    "coverage_percent": round(
                        percentage,
                        2,
                    ),
                })

    if not coverage_values:
        raise HTTPException(
            status_code=422,
            detail="No disaster demand has been reported",
        )

    return {
        "allocations": allocations,
        "remaining_stock_if_approved": stock,
        "ward_results": ward_results,
        "minimum_coverage_percent": round(
            min(coverage_values),
            2,
        ),
        "average_coverage_percent": round(
            sum(coverage_values)
            / len(coverage_values),
            2,
        ),
        "contains_allocations": bool(allocations),
    }


@router.get("/state")
def get_rolling_state():
    with LOCK:
        state = load_state()

        return {
            **state,
            "automatically_dispatches_resources": False,
            "note": (
                "All reservations and inventory "
                "represent simulation state only."
            ),
        }


@router.post("/reports")
def add_reports(payload: PipelineRequest):
    """
    Record incoming demand, retaining all
    previously approved reservations.
    """

    with LOCK:
        state = load_state()

        new_ids = [
            report.report_id
            for report in payload.reports
        ]

        old_ids = set(state["report_ids"])

        duplicates = sorted(
            old_ids.intersection(new_ids)
        )

        if duplicates:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Reports already recorded",
                    "report_ids": duplicates,
                },
            )

        incoming, processed = estimate_demand(
            payload.reports
        )

        for ward in WARDS:
            key = str(ward)

            for resource in RESOURCES:
                state["demand"][key][resource] += (
                    incoming[ward][resource]
                )

        state["report_ids"].extend(new_ids)

        state["report_history"].append({
            "received_at": now(),
            "reports": processed,
        })

        state["version"] += 1

        # A new report makes any earlier
        # unapproved proposal outdated.
        state["pending_review"] = None

        validate_state(state)
        save_state(state)

        return {
            "status": "reports_recorded",
            "reports_added": len(processed),
            "total_reports": len(state["report_ids"]),
            "version": state["version"],
            "demand": state["demand"],
            "reserved": state["reserved"],
            "requires_new_review": True,
            "automatically_dispatches_resources": False,
        }


@router.post("/review")
def review_rolling_allocation():
    """
    Calculate a new proposal.

    Does not reserve or dispatch anything.
    """

    with LOCK:
        state = load_state()

        # An unchanged proposal can be reviewed
        # repeatedly without creating new IDs.
        pending = state["pending_review"]

        if (
            pending is not None
            and pending["expected_version"]
            == state["version"]
        ):
            return pending

        proposal = build_proposal(state)

        review = {
            "review_id": str(uuid.uuid4()),
            "expected_version": state["version"],
            "created_at": now(),
            "method": (
                "Coverage-first fairness heuristic"
            ),
            "status": "awaiting_human_approval",
            "requires_human_approval": True,
            "automatically_dispatches_resources": False,
            **proposal,
        }

        if not proposal["contains_allocations"]:
            review["status"] = "no_resources_to_allocate"

        state["pending_review"] = review

        save_state(state)

        return review


@router.post(
    "/approve",
    dependencies=[Depends(require_approval_key)],
)
def approve_rolling_allocation(
    payload: ApprovalRequest,
):
    """
    Commit an approved reservation to
    persistent simulated inventory.

    This is NOT a real-world dispatch.
    """

    with LOCK:
        state = load_state()
        pending = state["pending_review"]

        if pending is None:
            raise HTTPException(
                status_code=409,
                detail="No pending allocation review",
            )

        if (
            payload.review_id != pending["review_id"]
            or payload.expected_version
            != state["version"]
            or pending["expected_version"]
            != state["version"]
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Stale or incorrect review. "
                    "Generate a new review."
                ),
            )

        if not pending["contains_allocations"]:
            raise HTTPException(
                status_code=409,
                detail="There is nothing to reserve",
            )

        allocations = pending["allocations"]

        # Check all operations before changing
        # the persistent state.
        stock_requirements = {}
        demand_requirements = {}

        for item in allocations:
            stock_key = (
                item["centre_id"],
                item["resource"],
            )

            demand_key = (
                str(item["ward_no"]),
                item["resource"],
            )

            quantity = item["quantity"]

            stock_requirements[stock_key] = (
                stock_requirements.get(
                    stock_key, 0
                ) + quantity
            )

            demand_requirements[demand_key] = (
                demand_requirements.get(
                    demand_key, 0
                ) + quantity
            )

        for (centre, resource), quantity in (
            stock_requirements.items()
        ):
            if (
                quantity
                > state["remaining_stock"][
                    centre
                ][resource]
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Insufficient simulated inventory",
                )

        for (ward, resource), quantity in (
            demand_requirements.items()
        ):
            available_need = (
                state["demand"][ward][resource]
                - state["reserved"][ward][resource]
            )

            if quantity > available_need:
                raise HTTPException(
                    status_code=409,
                    detail="Proposal exceeds unmet demand",
                )

        for item in allocations:
            centre = item["centre_id"]
            ward = str(item["ward_no"])
            resource = item["resource"]
            quantity = item["quantity"]

            state["remaining_stock"][
                centre
            ][resource] -= quantity

            state["reserved"][
                ward
            ][resource] += quantity

        approval = {
            "review_id": pending["review_id"],
            "approved_at": now(),
            "allocations": allocations,
            "simulation_only": True,
            "physical_dispatch_confirmed": False,
        }

        state["approval_history"].append(
            approval
        )

        state["version"] += 1
        state["pending_review"] = None

        validate_state(state)
        save_state(state)

        return {
            "status": "simulation_reservation_approved",
            "version": state["version"],
            "review_id": approval["review_id"],
            "allocations": allocations,
            "remaining_stock": state[
                "remaining_stock"
            ],
            "reserved": state["reserved"],
            "physical_dispatch_confirmed": False,
            "automatically_dispatches_resources": False,
        }


@router.post(
    "/reset",
    dependencies=[Depends(require_approval_key)],
)
def reset_rolling_simulation(
    payload: ResetRequest,
):
    """
    Development-only reset of the simulated
    rolling allocation ledger.
    """

    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to reset",
        )

    with LOCK:
        state = create_initial_state()
        save_state(state)

        return {
            "status": "simulation_reset",
            "version": 0,
            "simulation_only": True,
        }
