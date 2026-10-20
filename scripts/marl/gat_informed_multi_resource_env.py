
"""
RESPONDA — GAT-Informed Multi-Resource Environment

Extends the existing three-ward environment with
precomputed GAT features.

Original observation: 38 features per centre
GAT additions:       3 wards x 8 features = 24
New observation:      62 features per centre

The original environment and PPO checkpoint
remain unchanged.
"""

import csv
from pathlib import Path

import numpy as np

from multi_resource_cooperative_env import (
    MultiResourceCooperativeEnv,
)


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "aluva"
    / "aluva_gat_embeddings.csv"
)


class GATInformedMultiResourceEnv(
    MultiResourceCooperativeEnv
):
    def __init__(self, embeddings_path=None):
        self.embeddings_path = Path(
            embeddings_path or DEFAULT_EMBEDDINGS
        )

        self.gat_feature_count = 8
        self.gat_ward_vectors = {}

        # Load graph features BEFORE calling super().
        # The parent calls reset() during initialization.
        self._load_gat_features()

        super().__init__()

        self.original_observation_size = 38

        self.observation_size = (
            self.original_observation_size
            + len(self.wards) * self.gat_feature_count
        )

        assert self.observation_size == 62
        assert self.action_size == 16

    def _load_gat_features(self):
        if not self.embeddings_path.is_file():
            raise FileNotFoundError(
                "GAT embedding file not found: "
                f"{self.embeddings_path}"
            )

        with self.embeddings_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            columns = reader.fieldnames or []

            if "ward_no" not in columns:
                raise ValueError(
                    "GAT CSV must contain ward_no"
                )

            feature_columns = [
                column
                for column in columns
                if column.startswith("gat_feature_")
            ]

            def feature_number(column):
                suffix = column.rsplit("_", 1)[-1]

                return (
                    int(suffix)
                    if suffix.isdigit()
                    else -1
                )

            feature_columns.sort(
                key=feature_number
            )

            if len(feature_columns) != 8:
                raise ValueError(
                    "Expected exactly 8 GAT feature "
                    "columns starting with gat_feature_. "
                    f"Found: {feature_columns}"
                )

            raw_vectors = {}

            for row in reader:
                ward = int(row["ward_no"])

                if ward in raw_vectors:
                    raise ValueError(
                        f"Duplicate GAT row: Ward {ward}"
                    )

                vector = np.array(
                    [
                        float(row[column])
                        for column in feature_columns
                    ],
                    dtype=np.float32,
                )

                if not np.isfinite(vector).all():
                    raise ValueError(
                        "Non-finite GAT values for "
                        f"Ward {ward}"
                    )

                raw_vectors[ward] = vector

        required_wards = (3, 5, 9)

        missing = [
            ward
            for ward in required_wards
            if ward not in raw_vectors
        ]

        if missing:
            raise ValueError(
                f"Missing GAT embeddings: {missing}"
            )

        # Use a consistent mean and standard
        # deviation calculated from the saved graph.
        # The same normalization must be reused
        # when training and evaluating this model.

        all_vectors = np.stack(
            list(raw_vectors.values())
        )

        mean = all_vectors.mean(axis=0)
        std = all_vectors.std(axis=0)

        std = np.where(
            std < 1e-6,
            1.0,
            std,
        )

        self.gat_mean = mean
        self.gat_std = std

        for ward in required_wards:
            normalized = (
                raw_vectors[ward] - mean
            ) / std

            self.gat_ward_vectors[ward] = (
                normalized.astype(np.float32)
            )

    def get_gat_context(self):
        """
        Fixed ward ordering is essential.

        Ward 3: 8 features
        Ward 5: 8 features
        Ward 9: 8 features
        """

        return np.concatenate(
            [
                self.gat_ward_vectors[ward]
                for ward in self.wards
            ]
        ).astype(np.float32)

    def get_observations(self):
        """
        Existing resource information + graph context.

        Every relief-centre agent sees the same
        three-ward graph context, but maintains
        its own stock-related observation.
        """

        original = super().get_observations()

        gat_context = self.get_gat_context()

        observations = {}

        for centre_id in self.centre_ids:
            combined = np.concatenate(
                [
                    original[centre_id],
                    gat_context,
                ]
            ).astype(np.float32)

            if combined.shape != (62,):
                raise ValueError(
                    "Incorrect GAT-informed "
                    "observation shape"
                )

            observations[centre_id] = combined

        return observations


if __name__ == "__main__":
    import torch

    from multi_resource_ppo_network import (
        MultiResourceCooperativePPO,
    )

    print("=" * 52)
    print("PHASE 23 — GAT + MARL INTEGRATION TEST")
    print("=" * 52)

    env = GATInformedMultiResourceEnv()

    observations = env.reset()

    print(
        "Wards:",
        env.wards,
    )

    print(
        "Relief centres:",
        env.centre_ids,
    )

    print(
        "Original observation size:",
        env.original_observation_size,
    )

    print(
        "GAT features added:",
        len(env.get_gat_context()),
    )

    print(
        "New observation size:",
        env.observation_size,
    )

    print(
        "Number of actions:",
        env.action_size,
    )

    assert env.observation_size == 62
    assert env.action_size == 16

    for centre_id, observation in (
        observations.items()
    ):
        assert observation.shape == (62,)

        assert np.isfinite(
            observation
        ).all()

    print("PASS: GAT observations have 62 features")

    # Instantiate a NEW model with 62 inputs.
    # Do not load the old 38-input checkpoint.

    model = MultiResourceCooperativePPO(
        observation_size=62,
        action_size=16,
        num_agents=3,
    )

    state = torch.tensor(
        np.stack(
            [
                observations[centre]
                for centre in env.centre_ids
            ]
        ),
        dtype=torch.float32,
    ).unsqueeze(0)

    masks = env.get_action_masks()

    mask_tensor = torch.tensor(
        np.stack(
            [
                masks[centre]
                for centre in env.centre_ids
            ]
        ),
        dtype=torch.bool,
    ).unsqueeze(0)

    with torch.no_grad():
        (
            actions,
            log_probs,
            entropy,
            values,
        ) = model.get_actions_and_value(
            state,
            action_masks=mask_tensor,
        )

    assert state.shape == (1, 3, 62)
    assert actions.shape == (1, 3)
    assert values.shape == (1,)

    assert torch.isfinite(
        log_probs
    ).all()

    assert torch.isfinite(
        values
    ).all()

    for i in range(3):
        chosen = int(actions[0, i])

        assert mask_tensor[
            0, i, chosen
        ].item()

    print("PASS: PPO accepts 62-feature observations")
    print("PASS: Action masking still works")
    print("PASS: Actor and critic return finite outputs")

    print()
    print(
        "SUCCESS: GAT-informed environment "
        "and new PPO architecture validated."
    )

    print()
    print(
        "NOTE: This PPO model has random weights. "
        "It must be trained before allocation use."
    )
