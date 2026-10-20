
from fastapi import APIRouter, HTTPException

from backend.pipeline import PipelineRequest, run_pipeline


router = APIRouter(
    prefix="/api/pipeline",
    tags=["Allocation Safety Review"],
)


def evaluate_method(method_name, result):
    """Check allocation results and calculate demand coverage."""

    if result is None:
        return None

    coverage_values = []
    total_requested = 0
    total_delivered = 0
    total_unmet = 0
    ward_results = []

    for ward in result["wards"]:
        ward_no = int(ward["ward_no"])

        for resource in ward["resources"]:
            requested = int(resource["demand"])
            delivered = int(resource["delivered"])
            unmet = int(resource["unmet"])

            if (
                requested < 0
                or delivered < 0
                or delivered > requested
                or unmet != requested - delivered
            ):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Invalid {method_name} allocation "
                        f"for Ward {ward_no}"
                    ),
                )

            if requested == 0:
                continue

            coverage = delivered / requested

            coverage_values.append(coverage)
            total_requested += requested
            total_delivered += delivered
            total_unmet += unmet

            ward_results.append({
                "ward_no": ward_no,
                "resource": resource["resource"],
                "requested": requested,
                "delivered": delivered,
                "unmet": unmet,
                "coverage_percent": round(
                    coverage * 100, 2
                ),
            })

    if not coverage_values:
        raise HTTPException(
            status_code=422,
            detail="No positive resource demand to evaluate",
        )

    return {
        "method": method_name,
        "minimum_coverage_percent": round(
            min(coverage_values) * 100, 2
        ),
        "average_coverage_percent": round(
            (
                sum(coverage_values)
                / len(coverage_values)
            ) * 100,
            2,
        ),
        "total_requested": total_requested,
        "total_delivered": total_delivered,
        "total_unmet": total_unmet,
        "fairness_gap": result.get("fairness_gap"),
        "ward_results": ward_results,
    }


def choose_method(baseline, ppo):
    """
    Compare ward-resource coverage.

    Important: different resource units cannot be
    combined into a meaningful physical quantity.

    Therefore, selection first compares the WORST
    individual ward-resource coverage percentage,
    then the average coverage percentage.
    """

    if ppo is None:
        return "baseline", (
            "PPO is unavailable. The baseline is the "
            "only evaluated allocation candidate."
        )

    def score(candidate):
        return (
            candidate["minimum_coverage_percent"],
            candidate["average_coverage_percent"],
        )

    baseline_score = score(baseline)
    ppo_score = score(ppo)

    if baseline_score > ppo_score:
        return "baseline", (
            "The fairness baseline provides better "
            "coverage of reported resource demand."
        )

    if ppo_score > baseline_score:
        return "ppo", (
            "PPO provides better coverage of "
            "reported resource demand."
        )

    # Equal coverage: use the existing baseline
    # until a stronger PPO advantage is demonstrated.
    return "baseline", (
        "Both methods achieve equal coverage. "
        "The fairness baseline is preferred until "
        "additional evaluation establishes an advantage."
    )


@router.post("/review")
def review_allocation(payload: PipelineRequest):
    # Execute the already-working three-ward pipeline.
    pipeline_result = run_pipeline(payload)

    baseline = evaluate_method(
        "baseline",
        pipeline_result["baseline"],
    )

    ppo = evaluate_method(
        "ppo",
        pipeline_result["ppo"],
    )

    selected, reason = choose_method(
        baseline,
        ppo,
    )

    candidates = {
        "baseline": baseline,
        "ppo": ppo,
    }

    selected_result = candidates[selected]

    warnings = []

    if selected_result["total_unmet"] > 0:
        warnings.append(
            "Some demand remains unmet. Review available "
            "stock and alternative relief sources."
        )

    if (
        ppo is not None
        and ppo["minimum_coverage_percent"]
        < baseline["minimum_coverage_percent"]
    ):
        warnings.append(
            "PPO produces a lower minimum coverage "
            "than the fairness baseline."
        )

    return {
        "status": "review_complete",
        "supported_wards": [3, 5, 9],
        "reports_processed": pipeline_result[
            "report_count"
        ],
        "recommended_method": selected,
        "recommendation_reason": reason,
        "comparison": candidates,
        "recommended_allocations": pipeline_result[
            selected
        ]["allocations"],
        "recommended_remaining_stock": pipeline_result[
            selected
        ]["remaining_stock"],
        "warnings": warnings,
        "requires_human_approval": True,
        "automatically_dispatches_resources": False,
        "limitations": (
            "Prototype allocation review using simulated "
            "inventory and extracted report demand. "
            "Selection prioritizes worst-case and average "
            "ward-resource coverage. It does not yet "
            "assess real travel times, operational safety, "
            "or fairness for vulnerable populations."
        ),
    }
