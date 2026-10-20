
"""
RESPONDA — Phase 26 + Phase 29
Rolling Allocation and Security Regression Tests

Tests:
01. Backend API connectivity
02. Persistent report integrity
03. Inventory conservation
04. Demand and reservation conservation
05. Approval history integrity
06. No automatic physical dispatch
07. Pending review integrity
08. Duplicate approval rejection
09. Unauthorized approval rejection
10. Unauthorized reset rejection

IMPORTANT:
- Does not create reports
- Does not approve valid reservations
- Does not reset the simulation
- Does not modify trained models
"""

import json
import os
import unittest

from urllib.request import Request, urlopen
from urllib.error import HTTPError


BASE_URL = "http://127.0.0.1:8000"

RESOURCES = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases",
]

WARDS = ["3", "5", "9"]


# ============================================================
# 1. HTTP REQUEST HELPER
# ============================================================

def request_json(
    path,
    method="GET",
    payload=None,
    headers=None,
):
    body = None

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request_headers = {
        "Content-Type": "application/json",
    }

    if headers:
        request_headers.update(headers)

    request = Request(
        BASE_URL + path,
        data=body,
        method=method,
        headers=request_headers,
    )

    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.load(response)

    except HTTPError as error:
        response_body = error.read().decode("utf-8")

        try:
            data = json.loads(response_body)
        except ValueError:
            data = {"detail": response_body}

        return error.code, data


# ============================================================
# 2. REGRESSION TESTS
# ============================================================

class TestRollingAllocation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        status, cls.state = request_json(
            "/api/rolling/state"
        )

        if status != 200:
            raise RuntimeError(
                f"Rolling API unavailable: HTTP {status}"
            )

    # --------------------------------------------------------
    # TEST 01 — API CONNECTIVITY
    # --------------------------------------------------------

    def test_01_api_connectivity(self):
        status, data = request_json(
            "/api/pipeline/status"
        )

        self.assertEqual(status, 200)

        self.assertEqual(
            set(data["supported_wards"]),
            {3, 5, 9},
        )

    # --------------------------------------------------------
    # TEST 02 — PERSISTENT REPORTS
    # --------------------------------------------------------

    def test_02_persistent_reports(self):
        state = self.state

        report_ids = state["report_ids"]

        # Report IDs must be unique.
        self.assertEqual(
            len(report_ids),
            len(set(report_ids)),
        )

        # Every recorded report should have
        # an entry in the report history.
        history_count = sum(
            len(entry["reports"])
            for entry in state["report_history"]
        )

        self.assertEqual(
            len(report_ids),
            history_count,
        )

    # --------------------------------------------------------
    # TEST 03 — INVENTORY CONSERVATION
    # --------------------------------------------------------

    def test_03_inventory_conservation(self):
        state = self.state

        for centre, initial_resources in (
            state["initial_stock"].items()
        ):
            for resource in RESOURCES:
                initial = initial_resources[resource]

                remaining = (
                    state["remaining_stock"]
                    [centre][resource]
                )

                approved = sum(
                    allocation["quantity"]
                    for approval in state[
                        "approval_history"
                    ]
                    for allocation in approval[
                        "allocations"
                    ]
                    if (
                        allocation["centre_id"] == centre
                        and allocation["resource"] == resource
                    )
                )

                # Initial inventory must equal
                # remaining stock + approved reservations.
                self.assertEqual(
                    initial,
                    remaining + approved,
                    f"Inventory mismatch: "
                    f"{centre}/{resource}",
                )

                self.assertGreaterEqual(
                    remaining,
                    0,
                )

    # --------------------------------------------------------
    # TEST 04 — DEMAND CONSERVATION
    # --------------------------------------------------------

    def test_04_demand_conservation(self):
        state = self.state

        for ward in WARDS:
            for resource in RESOURCES:
                demand = (
                    state["demand"]
                    [ward][resource]
                )

                reserved = (
                    state["reserved"]
                    [ward][resource]
                )

                # Reservations cannot exceed demand.
                self.assertGreaterEqual(
                    demand,
                    reserved,
                )

                self.assertGreaterEqual(
                    reserved,
                    0,
                )

                approved = sum(
                    allocation["quantity"]
                    for approval in state[
                        "approval_history"
                    ]
                    for allocation in approval[
                        "allocations"
                    ]
                    if (
                        str(allocation["ward_no"]) == ward
                        and allocation["resource"] == resource
                    )
                )

                # The reservation ledger must match
                # all historical approved allocations.
                self.assertEqual(
                    reserved,
                    approved,
                    f"Reservation mismatch: "
                    f"Ward {ward}/{resource}",
                )

    # --------------------------------------------------------
    # TEST 05 — APPROVAL HISTORY
    # --------------------------------------------------------

    def test_05_approval_history(self):
        approvals = self.state[
            "approval_history"
        ]

        approval_ids = [
            approval["review_id"]
            for approval in approvals
        ]

        # No review may be approved twice.
        self.assertEqual(
            len(approval_ids),
            len(set(approval_ids)),
        )

        for approval in approvals:
            self.assertTrue(
                approval["simulation_only"]
            )

            self.assertFalse(
                approval["physical_dispatch_confirmed"]
            )

    # --------------------------------------------------------
    # TEST 06 — NO AUTOMATIC DISPATCH
    # --------------------------------------------------------

    def test_06_no_automatic_dispatch(self):
        state = self.state

        self.assertTrue(
            state["simulation_only"]
        )

        self.assertFalse(
            state["automatically_dispatches_resources"]
        )

    # --------------------------------------------------------
    # TEST 07 — PENDING REVIEW VALIDATION
    # --------------------------------------------------------

    def test_07_pending_review(self):
        state = self.state

        review = state["pending_review"]

        if review is None:
            return

        # The review must be based on the
        # current inventory version.
        self.assertEqual(
            review["expected_version"],
            state["version"],
        )

        for allocation in review["allocations"]:
            centre = allocation["centre_id"]
            ward = str(allocation["ward_no"])
            resource = allocation["resource"]
            quantity = allocation["quantity"]

            self.assertGreater(
                quantity,
                0,
            )

            # Sum all proposed allocations from
            # the same centre for the same resource.
            proposed_from_centre = sum(
                item["quantity"]
                for item in review["allocations"]
                if (
                    item["centre_id"] == centre
                    and item["resource"] == resource
                )
            )

            remaining_stock = (
                state["remaining_stock"]
                [centre][resource]
            )

            self.assertLessEqual(
                proposed_from_centre,
                remaining_stock,
            )

            # Check that total proposed quantities
            # do not exceed unmet ward demand.
            proposed_to_ward = sum(
                item["quantity"]
                for item in review["allocations"]
                if (
                    str(item["ward_no"]) == ward
                    and item["resource"] == resource
                )
            )

            unmet_demand = (
                state["demand"][ward][resource]
                - state["reserved"][ward][resource]
            )

            self.assertLessEqual(
                proposed_to_ward,
                unmet_demand,
            )

    # --------------------------------------------------------
    # TEST 08 — DUPLICATE APPROVAL REJECTION
    # --------------------------------------------------------

    def test_08_reject_duplicate_approval(self):
        approvals = self.state[
            "approval_history"
        ]

        if not approvals:
            self.skipTest(
                "No previously approved review exists."
            )

        approval_key = os.getenv(
            "RESPONDA_APPROVAL_KEY"
        )

        self.assertTrue(
            approval_key,
            "RESPONDA_APPROVAL_KEY must be set "
            "in the test terminal.",
        )

        historical_review_id = (
            approvals[0]["review_id"]
        )

        # Attempt to approve an already approved
        # review using valid authorization.
        status, _ = request_json(
            "/api/rolling/approve",
            method="POST",
            payload={
                "review_id": historical_review_id,
                "expected_version": self.state["version"],
            },
            headers={
                "X-Approval-Key": approval_key,
            },
        )

        self.assertEqual(
            status,
            409,
            "Duplicate approval must be rejected.",
        )

        # Confirm that the rejected request
        # did not change the saved inventory.
        latest_status, latest = request_json(
            "/api/rolling/state"
        )

        self.assertEqual(
            latest_status,
            200,
        )

        self.assertEqual(
            latest["version"],
            self.state["version"],
        )

        self.assertEqual(
            latest["remaining_stock"],
            self.state["remaining_stock"],
        )

        self.assertEqual(
            latest["approval_history"],
            self.state["approval_history"],
        )

    # --------------------------------------------------------
    # TEST 09 — UNAUTHORIZED APPROVAL
    # --------------------------------------------------------

    def test_09_reject_unauthorized_approval(self):
        # Deliberately omit the approval key.
        status, _ = request_json(
            "/api/rolling/approve",
            method="POST",
            payload={
                "review_id": "invalid-review",
                "expected_version": 0,
            },
        )

        self.assertEqual(
            status,
            401,
            "Approval without authorization "
            "must return HTTP 401.",
        )

        # Verify no inventory was modified.
        latest_status, latest = request_json(
            "/api/rolling/state"
        )

        self.assertEqual(
            latest_status,
            200,
        )

        self.assertEqual(
            latest["version"],
            self.state["version"],
        )

        self.assertEqual(
            latest["remaining_stock"],
            self.state["remaining_stock"],
        )

    # --------------------------------------------------------
    # TEST 10 — UNAUTHORIZED RESET
    # --------------------------------------------------------

    def test_10_reject_unauthorized_reset(self):
        # IMPORTANT:
        # confirm=False ensures this test cannot
        # reset the ledger even if authentication
        # is accidentally misconfigured.
        status, _ = request_json(
            "/api/rolling/reset",
            method="POST",
            payload={
                "confirm": False,
            },
        )

        self.assertEqual(
            status,
            401,
            "Reset without authorization "
            "must return HTTP 401.",
        )

        # Verify simulation state is unchanged.
        latest_status, latest = request_json(
            "/api/rolling/state"
        )

        self.assertEqual(
            latest_status,
            200,
        )

        self.assertEqual(
            latest["version"],
            self.state["version"],
        )

        self.assertEqual(
            latest["remaining_stock"],
            self.state["remaining_stock"],
        )

        self.assertEqual(
            latest["report_ids"],
            self.state["report_ids"],
        )


# ============================================================
# 3. RUN TEST SUITE
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESPONDA — REGRESSION AND SECURITY TESTS")
    print("PHASE 26 + PHASE 29")
    print("=" * 60)

    unittest.main(verbosity=2)
