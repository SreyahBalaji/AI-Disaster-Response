
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

BASE = Path("data/processed/aluva")

wards = gpd.read_file(BASE / "aluva_wards_clean.geojson")
features = pd.read_csv(BASE / "aluva_ward_features.csv")
edges = pd.read_csv(BASE / "aluva_graph_edges.csv")
flood = pd.read_csv(BASE / "aluva_flood_exposure.csv")

ward_ids = set(wards["Ward_No"].astype(int))
feature_ids = set(features["ward_no"].astype(int))

assert len(wards) == 26
assert len(features) == 26
assert ward_ids == feature_ids == set(range(1, 27))

assert wards.geometry.notna().all()
assert wards.geometry.is_valid.all()

assert not features["ward_no"].duplicated().any()
assert features.isna().sum().sum() == 0

assert features["population_2011_estimated"].sum() == 22428

periods = [10, 25, 50, 100, 200, 500]

assert len(flood) == 156
assert set(flood["return_period_years"]) == set(periods)

assert not flood.duplicated(
    ["ward_no", "return_period_years"]
).any()

assert flood["flood_exposure_proxy"].between(0, 1).all()

graph = nx.Graph()
graph.add_nodes_from(ward_ids)

for _, edge in edges.iterrows():
    source = int(edge["source_ward"])
    target = int(edge["target_ward"])

    assert source in ward_ids
    assert target in ward_ids
    assert source != target
    assert edge["shared_boundary_m"] > 1

    graph.add_edge(source, target)

assert graph.number_of_nodes() == 26
assert graph.number_of_edges() == 49
assert nx.is_connected(graph)

print("Ward boundaries: PASS")
print("Ward feature alignment: PASS")
print("Population total: PASS")
print("Flood exposure records: PASS")
print("Graph connectivity: PASS")

print("\nPHASE 2 VALIDATION SUCCESSFUL")
