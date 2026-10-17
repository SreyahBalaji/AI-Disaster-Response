
from cooperative_allocation_env import CooperativeAllocationEnv

RESOURCE = "water_litres"

env = CooperativeAllocationEnv()
env.dynamic_demand_enabled = True

observations = env.reset()

total_reward = 0.0

while True:
    actions = {}

    # Each centre chooses a ward based on current
    # unmet demand and demand-fulfilment ratios.
    # This is a simple centralized heuristic.
    simulated_demand = {
        ward: env.env.demand[ward][RESOURCE]
        for ward in env.wards
    }

    simulated_delivered = {
        ward: env.env.delivered[ward][RESOURCE]
        for ward in env.wards
    }

    for centre_id in env.centre_ids:
        centre = next(
            c for c in env.env.centres
            if c["centre_id"] == centre_id
        )

        stock = centre["resources"][RESOURCE]

        if stock <= 0:
            actions[centre_id] = 3
            continue

        eligible = [
            ward for ward in env.wards
            if simulated_demand[ward] > 0
        ]

        if not eligible:
            actions[centre_id] = 3
            continue

        def fulfilment_ratio(ward):
            original = (
                simulated_demand[ward]
                + simulated_delivered[ward]
            )
            return simulated_delivered[ward] / original

        chosen_ward = min(
            eligible,
            key=lambda ward: (
                fulfilment_ratio(ward),
                ward
            )
        )

        actions[centre_id] = env.wards.index(chosen_ward)

        amount = min(
            10,
            stock,
            simulated_demand[chosen_ward]
        )

        simulated_demand[chosen_ward] -= amount
        simulated_delivered[chosen_ward] += amount

    observations, rewards, terminated, truncated, info = (
        env.step(actions)
    )

    total_reward += rewards[env.centre_ids[0]]

    if terminated or truncated:
        break


deliveries = {
    ward: env.env.delivered[ward][RESOURCE]
    for ward in env.wards
}

ratios = []

for ward in env.wards:
    delivered = deliveries[ward]
    remaining = env.env.demand[ward][RESOURCE]
    original = delivered + remaining

    ratios.append(delivered / original)

total_delivered = sum(deliveries.values())

print("\nTIME-CONSTRAINED DYNAMIC BASELINE")
print("--------------------------------")
print("Steps:", env.steps)
print("Ward deliveries:", deliveries)
print("Total delivered:", total_delivered)
print("Demand satisfaction:", round(total_delivered / 1496, 4))
print("Fairness gap:", round(max(ratios) - min(ratios), 4))
print("Total team reward:", round(total_reward, 4))

print("\nCooperative PPO averages:")
print("Delivered: 1001.50 L")
print("Fairness gap: 0.6906")

print("\nSUCCESS! Time-constrained baseline evaluated.")
