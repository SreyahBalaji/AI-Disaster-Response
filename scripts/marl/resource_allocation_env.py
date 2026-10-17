
import json
from pathlib import Path


class ResourceAllocationEnv:

    def __init__(self):
        base = Path("data/processed")

        with open(
            base / "aluva/aluva_relief_centres.json",
            "r",
            encoding="utf-8"
        ) as file:
            self.initial_centres = json.load(file)

        with open(
            base / "crisis_reports/aluva_cumulative_demand.json",
            "r",
            encoding="utf-8"
        ) as file:
            self.snapshots = json.load(file)

        self.resource_types = [
            "water_litres",
            "food_meals",
            "rescue_cases",
            "shelter_households",
            "medical_cases"
        ]

        self.reset()

    def reset(self):
        self.centres = json.loads(
            json.dumps(self.initial_centres)
        )

        self.demand = {
            int(ward): resources.copy()
            for ward, resources in
            self.snapshots[-1]["ward_demand"].items()
        }

        self.delivered = {
            ward: {
                resource: 0
                for resource in self.resource_types
            }
            for ward in self.demand
        }

        return self.get_state()

    def get_state(self):
        return {
            "centres": self.centres,
            "remaining_demand": self.demand,
            "delivered": self.delivered
        }

    def allocate(self, centre_id, ward_no, resource, quantity):
        centre = next(
            (
                c for c in self.centres
                if c["centre_id"] == centre_id
            ),
            None
        )

        if centre is None:
            raise ValueError("Invalid centre ID")

        if ward_no not in self.demand:
            raise ValueError("Invalid ward number")

        if resource not in self.resource_types:
            raise ValueError("Invalid resource type")

        if type(quantity) is not int or quantity <= 0:
            raise ValueError(
                "Quantity must be a positive integer"
            )

        available = centre["resources"][resource]
        needed = self.demand[ward_no][resource]

        allocated = min(quantity, available, needed)

        centre["resources"][resource] -= allocated
        self.demand[ward_no][resource] -= allocated
        self.delivered[ward_no][resource] += allocated

        return allocated
