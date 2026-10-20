
from multi_resource_cooperative_env import MultiResourceCooperativeEnv


def run_baseline(dynamic=False):
    env = MultiResourceCooperativeEnv()
    env.dynamic_demand_enabled = dynamic
    env.reset()

    while True:
        actions = {}

        for centre_id in env.centre_ids:
            masks = env.get_action_masks()[centre_id]

            candidates = []

            for action, (ward, resource) in enumerate(env.action_map):
                if not masks[action]:
                    continue

                delivered = env.env.delivered[ward][resource]
                remaining = env.env.demand[ward][resource]
                total = delivered + remaining

                satisfaction = (
                    delivered / total if total > 0 else 1.0
                )

                # Prioritize low satisfaction and vulnerability.
                priority = (
                    (1.0 - satisfaction)
                    * env.vulnerability[ward]
                )

                candidates.append((priority, action))

            if candidates:
                candidates.sort(reverse=True)
                actions[centre_id] = candidates[0][1]
            else:
                actions[centre_id] = env.wait_action

        _, _, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            break

    print("\n" + "=" * 50)
    print("FAIRNESS BASELINE:", "DYNAMIC" if dynamic else "STATIC")
    print("=" * 50)

    print("Simulation steps:", env.steps)
    print("Fairness gap:", round(info["fairness_gap"], 4))

    for resource in env.resources:
        demand = sum(
            env.env.demand[ward][resource]
            + env.env.delivered[ward][resource]
            for ward in env.wards
        )

        delivered = sum(
            env.env.delivered[ward][resource]
            for ward in env.wards
        )

        if demand > 0:
            satisfaction = 100 * delivered / demand
            print(
                f"{resource}: {satisfaction:.2f}% satisfied"
            )
        else:
            print(f"{resource}: N/A (no demand)")

    print("\nWard 3 water delivered:",
          env.env.delivered[3]["water_litres"])

    return env


if __name__ == "__main__":
    run_baseline(dynamic=False)
    run_baseline(dynamic=True)

    print("\nSUCCESS: Fairness baseline evaluation completed.")
