
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

BASE = Path("data/processed/aluva")

FEATURES = BASE / "aluva_ward_features.csv"
EDGES = BASE / "aluva_graph_edges.csv"
OUTPUT = BASE / "aluva_gat_data.pt"

wards = pd.read_csv(FEATURES)
edges = pd.read_csv(EDGES)

wards = wards.sort_values("ward_no").reset_index(drop=True)

feature_columns = [
    "population_2011_estimated",
    "area_km2",
    "flood_exposure_10yr",
    "flood_exposure_25yr",
    "flood_exposure_50yr",
    "flood_exposure_100yr",
    "flood_exposure_200yr",
    "flood_exposure_500yr"
]

assert len(wards) == 26
assert wards["ward_no"].is_unique
assert not wards[feature_columns].isnull().any().any()

# Convert ward numbers into zero-based graph indices.
ward_to_index = {
    int(ward): index
    for index, ward in enumerate(wards["ward_no"])
}

# Normalize features using statistics from this graph.
values = torch.tensor(
    wards[feature_columns].to_numpy(),
    dtype=torch.float32
)

mean = values.mean(dim=0)
std = values.std(dim=0, unbiased=False).clamp(min=1e-8)

x = (values - mean) / std

edge_pairs = []

for _, edge in edges.iterrows():
    source = ward_to_index[int(edge["source_ward"])]
    target = ward_to_index[int(edge["target_ward"])]

    # Undirected adjacency represented in both directions.
    edge_pairs.append([source, target])
    edge_pairs.append([target, source])

edge_index = torch.tensor(
    edge_pairs,
    dtype=torch.long
).t().contiguous()

graph = Data(
    x=x,
    edge_index=edge_index
)

graph.ward_numbers = torch.tensor(
    wards["ward_no"].to_numpy(),
    dtype=torch.long
)

graph.feature_names = feature_columns
graph.feature_mean = mean
graph.feature_std = std

assert graph.num_nodes == 26
assert graph.edge_index.shape[1] == 98
assert torch.isfinite(graph.x).all()

torch.save(graph, OUTPUT)

print("Graph nodes:", graph.num_nodes)
print("Node features:", graph.num_node_features)
print("Directed edges:", graph.num_edges)
print("Node tensor shape:", tuple(graph.x.shape))
print("Edge tensor shape:", tuple(graph.edge_index.shape))

print("\nSUCCESS! Saved:", OUTPUT)
