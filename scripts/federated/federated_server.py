
import torch


def federated_average(local_models):
    """
    Aggregate model parameters using Federated Averaging.
    Assumes all models have the same architecture.
    """

    if not local_models:
        raise ValueError("No local models provided")

    global_model = {}

    for key in local_models[0]:
        parameters = [
            model[key].float()
            for model in local_models
        ]

        global_model[key] = torch.stack(parameters).mean(dim=0)

    return global_model


if __name__ == "__main__":

    print("FEDERATED LEARNING TEST")
    print("----------------------------")

    centre_1 = {
        "weight": torch.tensor([1.0, 3.0])
    }

    centre_2 = {
        "weight": torch.tensor([3.0, 5.0])
    }

    centre_3 = {
        "weight": torch.tensor([5.0, 7.0])
    }

    global_model = federated_average([
        centre_1,
        centre_2,
        centre_3
    ])

    print("Global model weights:", global_model["weight"])

    assert torch.allclose(
        global_model["weight"],
        torch.tensor([3.0, 5.0])
    )

    print("\nSUCCESS! Federated averaging verified.")
