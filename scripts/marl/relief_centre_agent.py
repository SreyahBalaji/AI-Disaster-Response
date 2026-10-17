
from resource_allocation_env import ResourceAllocationEnv


class ReliefCentreAgent:
    def __init__(self, centre_id):
        self.centre_id = centre_id

    def act(self, env, ward_no, resource, quantity):
        return env.allocate(
            centre_id=self.centre_id,
            ward_no=ward_no,
            resource=resource,
            quantity=quantity
        )


if __name__ == "__main__":
    env = ResourceAllocationEnv()

    agents = [
        ReliefCentreAgent(centre["centre_id"])
        for centre in env.centres
    ]

    print("MULTI-AGENT ENVIRONMENT")
    print("-----------------------------")
    print("Total agents:", len(agents))

    for agent in agents:
        allocated = agent.act(
            env,
            ward_no=3,
            resource="water_litres",
            quantity=50
        )

        print(
            f"{agent.centre_id} allocated "
            f"{allocated} litres"
        )

    total_delivered = sum(
        ward["water_litres"]
        for ward in env.delivered.values()
    )

    print("\nTotal water delivered:", total_delivered)

    assert total_delivered == 150

    print("\nSUCCESS! Three agents allocated resources.")
