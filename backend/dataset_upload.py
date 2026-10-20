
"""
RESPONDA - Dataset Upload and Analysis

Accepts CSV/JSON disaster reports.
Validates reports before allocation analysis.

Uploaded data is processed in memory.
No database changes, model training, or real dispatch.
"""

import csv
import io
import json
import re

from pathlib import PurePath

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from backend.pipeline import (
    PipelineRequest,
    ReportInput,
    estimate_demand,
)

from backend.allocation_safety import review_allocation


router = APIRouter(
    prefix="/api/datasets",
    tags=["Dataset Analysis"],
)

MAX_CHARACTERS = 1_000_000
MAX_REPORTS = 100

REQUIRED_COLUMNS = {
    "report_id",
    "detected_ward_no",
    "requested_resources",
}

FIELDS = {
    "report_id",
    "detected_ward_no",
    "affected_families",
    "affected_patients",
    "requested_resources",
    "text",
}


class DatasetSubmission(BaseModel):
    filename: str = Field(
        min_length=1,
        max_length=200,
    )

    content: str = Field(
        min_length=1,
        max_length=MAX_CHARACTERS,
    )


def parse_resources(value):
    """
    Accept:
      water;rescue
      water,food
      ["water", "rescue"]
    """

    if isinstance(value, list):
        return [
            str(item).strip().lower()
            for item in value
        ]

    if not isinstance(value, str):
        return value

    value = value.strip()

    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Invalid JSON resource list"
            ) from error

        if not isinstance(parsed, list):
            raise ValueError(
                "requested_resources must be a list"
            )

        return parse_resources(parsed)

    return [
        item.strip().lower()
        for item in re.split(
            r"[;,|]",
            value,
        )
        if item.strip()
    ]


def parse_dataset(submission: DatasetSubmission):
    """
    Convert uploaded CSV/JSON into validated
    three-ward disaster reports.
    """

    extension = PurePath(
        submission.filename
    ).suffix.lower()

    content = submission.content.lstrip(
        "\ufeff"
    )

    if extension not in {".csv", ".json"}:
        raise HTTPException(
            status_code=422,
            detail="Upload a .csv or .json file",
        )

    if extension == ".json":
        try:
            parsed = json.loads(content)

        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid JSON: {error.msg}"
                ),
            ) from error

        rows = (
            parsed.get("reports")
            if isinstance(parsed, dict)
            else parsed
        )

        if not isinstance(rows, list):
            raise HTTPException(
                status_code=422,
                detail=(
                    "JSON must contain an array "
                    'or {"reports": [...]}'
                ),
            )

    else:
        try:
            reader = csv.DictReader(
                io.StringIO(
                    content,
                    newline="",
                )
            )

            columns = set(
                reader.fieldnames or []
            )

            if not REQUIRED_COLUMNS.issubset(
                columns
            ):
                missing = sorted(
                    REQUIRED_COLUMNS - columns
                )

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Missing CSV columns: "
                        + ", ".join(missing)
                    ),
                )

            rows = list(reader)

        except csv.Error as error:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid CSV: {error}",
            ) from error

    if not 1 <= len(rows) <= MAX_REPORTS:
        raise HTTPException(
            status_code=422,
            detail=(
                "Dataset must contain between "
                f"1 and {MAX_REPORTS} reports"
            ),
        )

    reports = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        if (
            not isinstance(row, dict)
            or None in row
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Row {index}: "
                    "invalid column structure"
                ),
            )

        normalized = {}

        for key in FIELDS:
            if key not in row:
                continue

            value = row[key]

            if isinstance(value, str):
                value = value.strip()

            if (
                key in {
                    "affected_families",
                    "affected_patients",
                }
                and value == ""
            ):
                value = None

            if key == "requested_resources":
                try:
                    value = parse_resources(
                        value
                    )

                except ValueError as error:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Row {index}: {error}"
                        ),
                    ) from error

            normalized[key] = value

        try:
            report = ReportInput.model_validate(
                normalized
            )

        except ValidationError as error:
            problem = error.errors()[0]

            field = ".".join(
                str(part)
                for part in problem["loc"]
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Row {index}, {field}: "
                    f"{problem['msg']}"
                ),
            ) from error

        reports.append(report)

    # Reuse the existing validated
    # NLP-to-demand conversion.
    demand, processed = estimate_demand(
        reports
    )

    has_demand = any(
        quantity > 0
        for ward in demand.values()
        for quantity in ward.values()
    )

    if not has_demand:
        raise HTTPException(
            status_code=422,
            detail=(
                "All resource demands are zero"
            ),
        )

    return reports, demand, processed


@router.post("/validate")
def validate_dataset(
    submission: DatasetSubmission,
):
    """
    Validate uploaded data without
    running allocation models.
    """

    reports, demand, processed = (
        parse_dataset(submission)
    )

    return {
        "valid": True,
        "file_name": submission.filename,
        "report_count": len(reports),
        "supported_wards": [3, 5, 9],
        "combined_demand": demand,
        "preview": processed[:10],
        "saved_to_database": False,
        "training_started": False,
    }


@router.post("/analyze")
def analyze_dataset(
    submission: DatasetSubmission,
):
    """
    Validate data, run both allocation
    methods, and return the safety review.
    """

    reports, demand, _ = (
        parse_dataset(submission)
    )

    review = review_allocation(
        PipelineRequest(
            reports=reports
        )
    )

    return {
        "file_name": submission.filename,
        "report_count": len(reports),
        "combined_demand": demand,
        "review": review,
        "saved_to_database": False,
        "training_started": False,
        "simulation_only": True,
    }
