
from pathlib import Path
import pandas as pd

from resource_allocation_env import ResourceAllocationEnv

env = ResourceAllocationEnv()

priority_path = Path(
    "data/processed/aluva/aluva_ward_priority.csv"
)

priority_df = pd.read_csv(priority_path)

records = []

for ward_no, resources in env.demand.items():
    water_demand = resources["water_litres"]

    if water_demand <= 0:
        continue

    ward_info = priority_df.loc[
        priority_df["ward_no"] == ward_no
    ]

    assert len(ward_info) == 1

    records.append({
        "ward_no": ward_no,
        "ward_name": ward_info.iloc[0]["ward_name"],
        "water_demand": water_demand,
        "flood_priority_score": ward_info.iloc[0][
            "flood_priority_score"
        ]
    })

result = pd.DataFrame(records).sort_values(
    "flood_priority_score",
    ascending=False
)

print("FLOOD PRIORITY + WATER DEMAND")
print("--------------------------------")
print(result.to_string(index=False))

print("\nSUCCESS! Priority and demand integrated.")
