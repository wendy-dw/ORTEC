from src.evaluate import Evaluate
from src.supply_chain_config import SupplyChainConfig
from src.policy import MeanDemandPolicy, StaticBaseStockPolicyRandom, StaticBaseStockPolicyShortfall
from src.dynamics import Dynamics


def main():
    # Create an instance of supply chain configuration
    supply_chain_config = SupplyChainConfig()
    dynamics = Dynamics(supply_chain_config)

    # Policy parameters
    hedging_base_stock_shortfall = 0.25
    hedging_base_stock_random = 0.25
    hedging_mean_demand_policy = 0.1

    # Get the policies
    base_stock_policy_shortfall = StaticBaseStockPolicyShortfall(supply_chain_config, hedging_base_stock_shortfall)
    base_stock_policy_random = StaticBaseStockPolicyRandom(supply_chain_config, hedging_base_stock_random)
    mean_demand_policy = MeanDemandPolicy(supply_chain_config, hedging_mean_demand_policy)

    # Parameters for testing
    num_trajectories = 1000
    periods_per_trajectory = 60

    # To test one policy
    evaluate = Evaluate(supply_chain_config, dynamics, num_trajectories, periods_per_trajectory)
    avg_costs = evaluate.evaluate_policy(base_stock_policy_shortfall)
    print(avg_costs)

    # To compare multiple policies
    policy_costs = evaluate.compare_policies([mean_demand_policy, base_stock_policy_shortfall, base_stock_policy_random])
    print(policy_costs)


if __name__ == "__main__":
    main()
