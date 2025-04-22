import random
import math
import numpy as np
import pickle
from src.state import State


class Dynamics:
    def __init__(self, supply_chain_config):
        self.supply_chain_config = supply_chain_config

    # One time step of the simulation
    def transition_dynamics(self, state, action, event=None):
        # Check if action is feasible
        if self.feasible_action(state, action):
            self.update_state_with_action(state, action)
        else:
            raise Exception("Action is not feasible")

        # Satisfy demand, if possible
        # If event is given as input, use that as demand
        if event:
            self.update_state_with_demand(state, event)
        # Otherwise, generate demand and satisfy, if possible
        else:
            demand = self.get_demand(state)
            self.update_state_with_demand(state, demand)

        # Calculate reward
        costs = self.get_costs(state)

        # Transition to new time period
        self.update_state_with_transition(state)

        return state, costs

    def update_state_with_action(self, state, action):
        for i in range(self.supply_chain_config.num_components):
            # Implement the production decision
            state.inventory_in_pipeline_per_node[i][-1] += action[i]
            #  Remove the production decision from the on-hand inventory at predecessors
            for j in self.supply_chain_config.predecessors[i]:
                state.inventory_per_node[j] -= action[i]

    def update_state_with_demand(self, state, demand):
        # For each product
        for i in range(self.supply_chain_config.num_products):
            # Subtract demand the inventory
            state.inventory_per_node[self.supply_chain_config.num_sub_components + i] -= demand[i]
            # Append the demand to history demand of state
            state.history_demand_per_product[i].append(demand[i])

        # Update aggregate demand history
        state.aggregate_demand_model_history = np.append(state.aggregate_demand_model_history, demand[-1])

    def get_costs(self, state):
        costs = 0
        # Calculate the holding costs for the components
        for i in range(self.supply_chain_config.num_sub_components):
            costs += state.inventory_per_node[i] * self.supply_chain_config.h[i]

        # Calculate the reward for the products
        for i in range(self.supply_chain_config.num_products):
            # If backlog
            if state.inventory_per_node[self.supply_chain_config.num_sub_components + i] < 0:
                costs += (-state.inventory_per_node[self.supply_chain_config.num_sub_components + i]
                          * self.supply_chain_config.p[i])
            # Holding costs
            else:
                costs += (state.inventory_per_node[self.supply_chain_config.num_sub_components + i]
                          * self.supply_chain_config.h[self.supply_chain_config.num_sub_components + i])

        return costs

    def update_state_with_transition(self, state):
        # Remove the oldest production decision from the inventory in pipeline
        for i in range(self.supply_chain_config.num_components):
            # add the arriving products to the inventory
            state.inventory_per_node[i] += state.inventory_in_pipeline_per_node[i][0]
            state.inventory_in_pipeline_per_node[i].pop(0)
            # Add the placeholder for the production in the next time slot
            state.inventory_in_pipeline_per_node[i].append(0)

    # Checks if the action is feasible in the current state
    # Returns false if the action is not feasible
    def feasible_action(self, state, action):
        feasible = True
        # Check if action does not exceed capacity group
        for i in range(self.supply_chain_config.num_nodes):
            if (sum([action[j] for j in self.supply_chain_config.capacity_groups_indices[i]])
                    > self.supply_chain_config.capacities[i]):
                feasible = False
                break

        # Check material availability
        for i in range(self.supply_chain_config.num_sub_components):
            if state.inventory_per_node[i] < sum([action[j] for j in self.supply_chain_config.successors[i]]):
                feasible = False
                break

        return feasible

    # This function returns the demand for this period,
    # which is one list containing the demand for each product and the aggregated demand
    # The (fractional) aggregated demand will be used to generate the demand for future periods
    def get_demand(self, state):
        # Get demand vector for AR model, i.e., most recent observations are first elements
        aggregate_demand_history_ar_model = state.aggregate_demand_model_history[
                                            -self.supply_chain_config.aggregate_demand_num_lags:][::-1]

        # Generate next period aggregate demand based on AR model
        aggregate_demand = ((np.sum(self.supply_chain_config.aggregate_demand_ar_params *
                                    (aggregate_demand_history_ar_model - self.supply_chain_config.aggregate_demand_mean))
                             + np.random.normal(loc=0, scale=self.supply_chain_config.aggregate_demand_error_sigma))
                            + self.supply_chain_config.aggregate_demand_mean)

        # Censor at zero
        aggregate_demand = max(aggregate_demand, 0)

        # Generate the stochastic split parameters
        samples = list(range(self.supply_chain_config.num_products))
        for i in range(self.supply_chain_config.num_products):
            samples[i] = np.random.gamma(shape=self.supply_chain_config.product_demand_split_kappa[i],
                                         scale=self.supply_chain_config.product_demand_split_theta[i])

        # Compute the demand per product
        demand = [0 for _ in range(self.supply_chain_config.num_products)]
        indices = list(range(self.supply_chain_config.num_products))
        random.shuffle(indices)
        for i in indices[:-1]:
            demand[i] += round((samples[i] / sum(samples)) * aggregate_demand)
        demand[indices[-1]] += round(aggregate_demand) - sum(demand)
        demand.append(aggregate_demand)

        return demand

    # Get the fixed initial state to start simulating the trajectories
    def get_initial_state(self):
        state = State()
        state.inventory_per_node = [[] for _ in range(self.supply_chain_config.num_components)]
        state.inventory_in_pipeline_per_node = [[] for _ in range(self.supply_chain_config.num_components)]
        for i in range(self.supply_chain_config.num_components):
            # Initial state has 20% safety stock above mean demand
            state.inventory_per_node[i] = math.ceil(self.supply_chain_config.avg_demand_per_comp[i]*1.2)
            for j in range(self.supply_chain_config.lead_times[i] - 1):
                state.inventory_in_pipeline_per_node[i].append(math.ceil(
                    self.supply_chain_config.avg_demand_per_comp[i]))
            # Placeholder for the production in the current time slot
            state.inventory_in_pipeline_per_node[i].append(0)

        # Aggregate demand history (last position is the most recent observation)
        with open('data/aggregate_demand_model_history.pickle', 'rb') as handle:
            state.aggregate_demand_model_history = pickle.load(handle)

        # Demand history per product (last position is the most recent observation)
        with open('data/demand_products_history.pickle', 'rb') as handle:
            state.history_demand_per_product = list(pickle.load(handle))

        return state
