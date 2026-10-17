
from pathlib import Path

import geopandas as gpd
import pandas as pd
import networkx as nx

INPUT = Path(
    "data/processed/aluva/aluva_wards_clean.geojson"
)

OUTPUT = Path(
    "data/processed/aluva/aluva_graph_edges.csv"
)

wards = gpd.read_file(INPUT)

assert len(wards) == 26

# Use a projected CRS so boundary lengths are in metres.
wards = wards.to_crs(wards.estimate_utm_crs())
wards = wards.sort_values("Ward_No").reset_index(drop=True)

edges = []

for i in range(len(wards)):
    for j in range(i + 1, len(wards)):

        geometry_a = wards.loc[i, "geometry"]
        geometry_b = wards.loc[j, "geometry"]

        shared_boundary = (
            geometry_a.boundary.intersection(
                geometry_b.boundary
            )
        )

        shared_length = shared_boundary.length

        # Ignore point-only contacts and tiny boundary artifacts.
        if shared_length > 1.0:
            edges.append({
                "source_ward": int(wards.loc[i, "Ward_No"]),
                "target_ward": int(wards.loc[j, "Ward_No"]),
                "shared_boundary_m": round(shared_length, 2)
            })

edges_df = pd.DataFrame(
    edges,
    columns=[
        "source_ward",
        "target_ward",
        "shared_boundary_m"
    ]
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
edges_df.to_csv(OUTPUT, index=False)

graph = nx.Graph()

graph.add_nodes_from(
    wards["Ward_No"].astype(int).tolist()
)

for edge in edges:
    graph.add_edge(
        edge["source_ward"],
        edge["target_ward"]
    )

print("Total ward nodes:", graph.number_of_nodes())
print("Total adjacency edges:", graph.number_of_edges())
print("Connected components:", nx.number_connected_components(graph))

print(
    "Is graph connected:",
    nx.is_connected(graph)
)

print(
    "Isolated wards:",
    list(nx.isolates(graph))
)

print("\nEdges:")
print(edges_df.to_string(index=False))

print("\nSUCCESS! Saved:", OUTPUT)
