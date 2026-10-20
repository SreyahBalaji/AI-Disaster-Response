
import json
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
API = "http://127.0.0.1:8000"

RESOURCES = [
    "water_litres",
    "food_meals",
    "rescue_cases",
    "shelter_households",
    "medical_cases",
]

SUPPORTED_PPO_WARDS = {3, 5, 9}


def get_json(endpoint, expected_status=200):
    url = f"{API}{endpoint}"

    try:
        with urlopen(url, timeout=120) as response:
            status = response.status
            body = json.load(response)

    except HTTPError as error:
        status = error.code
        body = json.loads(
            error.read().decode("utf-8")
        )

    except URLError as error:
        raise AssertionError(
            f"Cannot connect to {url}: {error}"
        )

    if status != expected_status:
        raise AssertionError(
            f"{endpoint}: expected HTTP "
            f"{expected_status}, got {status}. "
            f"Response: {body}"
        )

    return body


def load_json(relative_path):
    path = DATA_DIR / relative_path

    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as file:
        return json.load(file)


class Phase18Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print("PHASE 18 — END-TO-END VALIDATION")
        print("=" * 60)

        cls.centres = load_json(
            "aluva/aluva_relief_centres.json"
        )

        cls.initial_stock = {
            centre["centre_id"]: {
                resource: int(
                    centre["resources"].get(resource, 0)
                )
                for resource in RESOURCES
            }
            for centre in cls.centres
        }

        cls.report_list = get_json(
            "/api/decision-support/reports"
        )["reports"]

        cls.results = {}

        for report in cls.report_list:
            report_id = report["report_id"]

            cls.results[report_id] = get_json(
                "/api/decision-support/recommend/"
                + report_id
            )

        print(
            f"Loaded {len(cls.results)} "
            "decision-support results."
        )

    def test_01_all_ten_reports(self):
        self.assertEqual(
            len(self.report_list),
            10
        )

        self.assertEqual(
            len(self.results),
            10
        )

        ids = [
            report["report_id"]
            for report in self.report_list
        ]

        self.assertEqual(
            len(ids),
            len(set(ids))
        )

        print("PASS: All ten reports evaluated")

    def test_02_report_and_ward_matching(self):
        for item in self.report_list:
            report_id = item["report_id"]
            result = self.results[report_id]

            with self.subTest(report=report_id):
                self.assertEqual(
                    result["report"]["report_id"],
                    report_id
                )

                self.assertEqual(
                    result["report"]["ward_no"],
                    item["ward_no"]
                )

                self.assertEqual(
                    result["ward"]["ward_no"],
                    item["ward_no"]
                )

        print("PASS: Reports match identified wards")

    def test_03_resource_conservation(self):
        for report_id, result in self.results.items():

            demand = result["demand"]
            recommendation = result["recommendation"]

            allocated_by_resource = defaultdict(int)
            allocated_by_centre = defaultdict(int)

            for allocation in recommendation["allocations"]:
                centre_id = allocation["centre_id"]
                resource = allocation["resource"]
                quantity = allocation["quantity"]

                with self.subTest(
                    report=report_id,
                    centre=centre_id,
                    resource=resource
                ):
                    self.assertIn(
                        centre_id,
                        self.initial_stock
                    )

                    self.assertIn(
                        resource,
                        RESOURCES
                    )

                    self.assertGreater(
                        quantity,
                        0
                    )

                allocated_by_resource[resource] += quantity

                allocated_by_centre[
                    (centre_id, resource)
                ] += quantity

            for resource in RESOURCES:
                with self.subTest(
                    report=report_id,
                    resource=resource
                ):
                    self.assertLessEqual(
                        allocated_by_resource[resource],
                        demand[resource]
                    )

            for (
                centre_id,
                resource
            ), quantity in allocated_by_centre.items():

                with self.subTest(
                    report=report_id,
                    centre=centre_id,
                    resource=resource
                ):
                    self.assertLessEqual(
                        quantity,
                        self.initial_stock[
                            centre_id
                        ][resource]
                    )

        print(
            "PASS: No allocation exceeds "
            "demand or initial stock"
        )

    def test_04_resource_summary_consistency(self):
        for report_id, result in self.results.items():

            allocations = result[
                "recommendation"
            ]["allocations"]

            summary = result[
                "recommendation"
            ]["resource_summary"]

            for item in summary:
                resource = item["resource"]

                allocated = sum(
                    allocation["quantity"]
                    for allocation in allocations
                    if allocation["resource"] == resource
                )

                with self.subTest(
                    report=report_id,
                    resource=resource
                ):
                    self.assertEqual(
                        item["requested"],
                        result["demand"][resource]
                    )

                    self.assertEqual(
                        item["allocated"],
                        allocated
                    )

                    self.assertEqual(
                        item["unmet"],
                        item["requested"] - allocated
                    )

                    self.assertGreaterEqual(
                        item["unmet"],
                        0
                    )

        print(
            "PASS: Requested, allocated and "
            "unmet values are consistent"
        )

    def test_05_remaining_stock_consistency(self):
        for report_id, result in self.results.items():

            allocations = result[
                "recommendation"
            ]["allocations"]

            remaining_stock = result[
                "recommendation"
            ]["remaining_stock"]

            for centre_id in self.initial_stock:
                for resource in RESOURCES:

                    allocated = sum(
                        allocation["quantity"]
                        for allocation in allocations
                        if (
                            allocation["centre_id"]
                            == centre_id
                            and allocation["resource"]
                            == resource
                        )
                    )

                    expected_remaining = (
                        self.initial_stock[
                            centre_id
                        ][resource]
                        - allocated
                    )

                    with self.subTest(
                        report=report_id,
                        centre=centre_id,
                        resource=resource
                    ):
                        self.assertEqual(
                            remaining_stock[
                                centre_id
                            ][resource],
                            expected_remaining
                        )

                        self.assertGreaterEqual(
                            expected_remaining,
                            0
                        )

        print("PASS: Remaining stock is correct")

    def test_06_ppo_compatibility(self):
        for report_id, result in self.results.items():

            ward = result["report"]["ward_no"]
            comparison = result["ppo_comparison"]

            with self.subTest(report=report_id):
                if ward in SUPPORTED_PPO_WARDS:
                    self.assertTrue(
                        comparison["available"],
                        comparison.get("reason")
                    )

                    self.assertEqual(
                        comparison["ward_no"],
                        ward
                    )

                    for algorithm in [
                        "ppo",
                        "baseline"
                    ]:
                        rows = comparison[
                            algorithm
                        ]["resources"]

                        for item in rows:
                            self.assertGreaterEqual(
                                item["allocated"],
                                0
                            )

                            self.assertLessEqual(
                                item["allocated"],
                                item["requested"]
                            )

                else:
                    self.assertFalse(
                        comparison["available"]
                    )

                    self.assertIn(
                        "supports Wards 3, 5 and 9",
                        comparison["reason"]
                    )

        print(
            "PASS: PPO compatibility and "
            "per-resource bounds are correct"
        )

    def test_07_gat_network(self):
        data = get_json(
            "/api/ai-insights/graph"
        )

        self.assertEqual(
            data["node_count"],
            26
        )

        self.assertEqual(
            data["edge_count"],
            49
        )

        self.assertEqual(
            data["centre_count"],
            3
        )

        self.assertEqual(
            data["total_visual_nodes"],
            29
        )

        node_ids = {
            node["id"]
            for node in data["nodes"]
        }

        for edge in data["edges"]:
            self.assertIn(
                edge["source"],
                node_ids
            )

            self.assertIn(
                edge["target"],
                node_ids
            )

        centre_ids = {
            centre["id"]
            for centre in data["relief_centres"]
        }

        for link in data["centre_links"]:
            self.assertIn(
                link["source"],
                centre_ids
            )

            self.assertIn(
                link["target"],
                node_ids
            )

        print(
            "PASS: GAT network and relief "
            "centre links are valid"
        )

    def test_08_nlp_api(self):
        data = get_json(
            "/api/ai-insights/nlp"
        )

        self.assertEqual(
            data["report_count"],
            10
        )

        self.assertEqual(
            data["method"],
            "rule_based_baseline"
        )

        self.assertEqual(
            data["source"],
            "synthetic"
        )

        print("PASS: NLP report API")

    def test_09_federated_api(self):
        data = get_json(
            "/api/ai-insights/federated"
        )

        evaluation = data["evaluation"]

        self.assertEqual(
            evaluation["aggregation_method"],
            "FedAvg"
        )

        self.assertEqual(
            evaluation["training_rounds"],
            5
        )

        self.assertEqual(
            len(
                evaluation["participating_centres"]
            ),
            3
        )

        self.assertEqual(
            len(data["training_history"]),
            5
        )

        print("PASS: Federated learning API")

    def test_10_existing_allocation_api(self):
        data = get_json(
            "/api/multi-resource/compare?dynamic=false"
        )

        self.assertIn("ppo", data)
        self.assertIn("baseline", data)

        for algorithm in ("ppo", "baseline"):
            self.assertEqual(
                len(
                    data[algorithm]["resources"]
                ),
                5
            )

        print("PASS: Existing MARL comparison API")

    def test_11_gis_api(self):
        data = get_json(
            "/api/gis/priority-map"
        )

        self.assertEqual(
            len(data["features"]),
            26
        )

        print("PASS: GIS priority map API")

    def test_12_known_report_r001(self):
        result = self.results["R001"]

        allocations = result[
            "recommendation"
        ]["allocations"]

        lookup = {
            (
                item["centre_id"],
                item["resource"]
            ): item["quantity"]
            for item in allocations
        }

        self.assertEqual(
            lookup[("RC01", "water_litres")],
            480
        )

        self.assertEqual(
            lookup[("RC01", "rescue_cases")],
            40
        )

        self.assertTrue(
            result["ppo_comparison"]["available"]
        )

        print("PASS: R001 reference scenario")

    def test_13_known_report_r004(self):
        result = self.results["R004"]

        allocations = result[
            "recommendation"
        ]["allocations"]

        self.assertTrue(
            any(
                item["centre_id"] == "RC03"
                and item["resource"] == "medical_cases"
                and item["quantity"] == 12
                for item in allocations
            )
        )

        self.assertFalse(
            result["ppo_comparison"]["available"]
        )

        print("PASS: R004 reference scenario")

    def test_14_unknown_report(self):
        response = get_json(
            "/api/decision-support/recommend/UNKNOWN",
            expected_status=404
        )

        self.assertIn(
            "detail",
            response
        )

        print("PASS: Unknown reports return HTTP 404")

    def test_15_explanations(self):
        required_sections = [
            "report_reason",
            "demand_reason",
            "ward_reason",
            "allocation_reason",
            "gat_note",
        ]

        for report_id, result in self.results.items():
            for section in required_sections:
                with self.subTest(
                    report=report_id,
                    section=section
                ):
                    self.assertTrue(
                        result["explanation"][section]
                    )

        print(
            "PASS: All reports contain "
            "explanation sections"
        )


if __name__ == "__main__":
    start = time.time()

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            Phase18Tests
        )
    )

    print("\n" + "=" * 60)
    print("PHASE 18 TEST SUMMARY")
    print("=" * 60)

    print("Tests run:", result.testsRun)
    print("Failures:", len(result.failures))
    print("Errors:", len(result.errors))
    print(
        "Duration:",
        round(time.time() - start, 2),
        "seconds"
    )

    if result.wasSuccessful():
        print(
            "SUCCESS: All automated checks passed."
        )
    else:
        print(
            "FAILED: Review the test errors above."
        )

    sys.exit(0 if result.wasSuccessful() else 1)
