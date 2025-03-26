class Evaluate:

    def __init__(self, supply_chain_config, dynamics, num_trajectories, periods_per_trajectory):
        self.supply_chain_config = supply_chain_config
        self.dynamics = dynamics
        self.num_trajectories = num_trajectories
        self.periods_per_trajectory = periods_per_trajectory

    def evaluate_policy(self, policy):
        policy_cost = []
        for i in range(self.num_trajectories):
            state = self.dynamics.get_initial_state()
            total_costs = 0
            for j in range(self.periods_per_trajectory):
                action = policy.set_action(state)
                state, cost = self.dynamics.transition_dynamics(state, action)
                total_costs += cost
            policy_cost.append(total_costs / self.periods_per_trajectory)

        avg_costs = sum(policy_cost) / self.num_trajectories

        return avg_costs

    def compare_policies(self, policies):
        policy_costs = []

        for i in range(self.num_trajectories):

            # Keep track of events for common random number between policies
            count = 0
            events = []
            for policy in policies:
                total_costs = 0
                state = self.dynamics.get_initial_state()

                for j in range(self.periods_per_trajectory):
                    if count == 0:
                        events.append(self.dynamics.get_demand(state))
                    action = policy.set_action(state)
                    state, cost = self.dynamics.transition_dynamics(state, action, events[j])
                    total_costs += cost

                avg_costs = total_costs / self.periods_per_trajectory
                policy_costs.append(avg_costs)
                count += 1

        policy_avg_costs = [sum(policy_costs[i::len(policies)]) / self.num_trajectories for i in range(len(policies))]

        return policy_avg_costs
