import random
import math


# Create your own policy
# 1) Create a class
# 2) Create a set_action function which takes state as input and returns the production quantity per node
# 3) Implement the logic of the policy in the set_action function

# Mean demand policy
# This policy always produces the mean demand
# Respecting material availability and capacity constraints
class MeanDemandPolicy:
    def __init__(self, supply_chain_config, hedging_parameter):
        self.supply_chain_config = supply_chain_config
        self.hedging_parameter = hedging_parameter

    def set_action(self, state):
        action_per_node = [0 for _ in range(self.supply_chain_config.num_components)]
        # Randomly order the components for production decision
        indices = list(range(self.supply_chain_config.num_components))
        random.shuffle(indices)
        for i in indices:
            # Produce the mean demand if possible
            action_per_node[i] += math.ceil(self.supply_chain_config.avg_demand_per_comp[i]*(1+self.hedging_parameter))
            # Check if capacity group is not exceeded
            if (sum([action_per_node[j] for j in self.supply_chain_config.capacity_groups_indices[
                self.supply_chain_config.capacity_groups[i]]])
                    > self.supply_chain_config.capacities[self.supply_chain_config.capacity_groups[i]]):
                action_per_node[i] -= (
                        sum([action_per_node[j] for j in self.supply_chain_config.capacity_groups_indices[
                            self.supply_chain_config.capacity_groups[i]]]) - self.supply_chain_config.capacities[
                            self.supply_chain_config.capacity_groups[i]])

            # Check if material availability is not exceeded
            for j in self.supply_chain_config.predecessors[i]:
                if (state.inventory_per_node[j]
                        < sum([action_per_node[k] for k in self.supply_chain_config.successors[j]])):
                    action_per_node[i] -= (sum([action_per_node[k] for k in self.supply_chain_config.successors[j]])
                                           - state.inventory_per_node[j])

        return action_per_node


# Static base-stock policy that minimizes expected shortfall
# This policy raises the echelon inventory position to the base-stock level
# Respecting material availability and capacity constraints
# With each allocation, it iteratively decides based on minimization of shortfall
class StaticBaseStockPolicyShortfall:
    def __init__(self, supply_chain_config, hedging_parameter):
        self.supply_chain_config = supply_chain_config
        self.hedging_parameter = hedging_parameter
        self.base_stock_levels = self.get_base_stock_levels()

        # Determine the parts after a divergent allocation decision
        self.divergent_components = [vector for vector in self.supply_chain_config.capacity_groups_indices if len(vector) > 1]
        filtered_divergent_components = []
        for sublist in self.divergent_components:
            if not any(set(sublist).issubset(set(other)) for other in self.divergent_components if sublist != other):
                filtered_divergent_components.append(sublist)
        self.divergent_components = filtered_divergent_components
        self.unique_divergent_components = list(set([item for sublist in self.divergent_components for item in sublist]))

    def set_action(self, state):
        action_per_node = [0 for _ in range(self.supply_chain_config.num_components)]

        # Randomly order the components for production decision
        indices = list(range(self.supply_chain_config.num_components))
        random.shuffle(indices)

        # Compute echelon inventory
        echelon_inventory = self.compute_echelon_inventory_position(state)

        # Decide production for each node
        while len(indices) > 0:
            # index i is the current node to decide for
            i = indices[0]

            # If part of allocation rule
            if i in self.unique_divergent_components:
                action_per_node = self.action_divergent_parts(state, echelon_inventory, i, action_per_node)

                indices = [element for element in indices if element not in [sublist for sublist in self.divergent_components if i in sublist][0]]
            else:
                # Raise the echelon inventory to its base-stock level if possible
                action_per_node[i] += self.base_stock_levels[i] - echelon_inventory[i]

                # Check if capacity group is not exceeded
                if (sum([action_per_node[j] for j in self.supply_chain_config.capacity_groups_indices[
                    self.supply_chain_config.capacity_groups[i]]])
                        > self.supply_chain_config.capacities[self.supply_chain_config.capacity_groups[i]]):
                    action_per_node[i] -= (
                            sum([action_per_node[j] for j in self.supply_chain_config.capacity_groups_indices[
                                self.supply_chain_config.capacity_groups[i]]]) - self.supply_chain_config.capacities[
                                self.supply_chain_config.capacity_groups[i]])

                # Check if material availability is not exceeded
                for j in self.supply_chain_config.predecessors[i]:
                    if (state.inventory_per_node[j]
                            < sum([action_per_node[k] for k in self.supply_chain_config.successors[j]])):
                        action_per_node[i] -= (sum([action_per_node[k] for k in self.supply_chain_config.successors[j]])
                                               - state.inventory_per_node[j])

                indices.pop(0)

        return action_per_node

    # Function to determine action for the allocation based on minimizing shortfall
    def action_divergent_parts(self, state, echelon_inventory, node, action_per_node):
        # Find other nodes to decide for in the divergent part
        nodes_allocation = [sublist.copy() for sublist in self.divergent_components if node in sublist][0]
        while True:
            # Compute the shortfall of those nodes
            expected_shortfall = self.compute_expected_shortfall(nodes_allocation, echelon_inventory)
            node_lowest_shortfall = nodes_allocation[expected_shortfall.index(max(expected_shortfall))]

            # Add the action to the node with the lowest shortfall
            # This will later be removed if the action is infeasible
            action_per_node[node_lowest_shortfall] += 1
            echelon_inventory[node_lowest_shortfall] += 1

            # Check if lowest shortfall is zero, i.e., all base-stock levels are reached
            if min(expected_shortfall) == 0:
                # Then stop iterating and return action as all nodes are at base-stock level
                action_per_node[node_lowest_shortfall] -= 1
                echelon_inventory[node_lowest_shortfall] -= 1
                break

            # If action is not feasible, try producing for other nodes
            if not self.check_action_feasibility(state, action_per_node, node_lowest_shortfall):
                action_per_node[node_lowest_shortfall] -= 1
                echelon_inventory[node_lowest_shortfall] -= 1
                nodes_allocation.remove(node_lowest_shortfall)

            # If there is no feasible action for all nodes, stop
            if len(nodes_allocation) == 0:
                break

        return action_per_node

    def check_action_feasibility(self, state, action_per_node, node):
        action = True
        # Check if materials are available for all predecessors of this node
        for j in self.supply_chain_config.predecessors[node]:
            if state.inventory_per_node[j] < sum([action_per_node[k] for k in self.supply_chain_config.successors[j]]):
                action = False
                break

        # Check if capacity group is not exceeded
        if (sum([action_per_node[j] for j in self.supply_chain_config.capacity_groups_indices[
            self.supply_chain_config.capacity_groups[node]]])
                > self.supply_chain_config.capacities[self.supply_chain_config.capacity_groups[node]]):
            action = False
        return action

    def compute_expected_shortfall(self, components, echelon_inventory):
        expected_shortfall = []
        for i in components:
            expected_shortfall.append(max(0, (self.base_stock_levels[i] - echelon_inventory[i])/self.base_stock_levels[i]))
        return expected_shortfall

    def compute_echelon_inventory_position(self, state):
        # Compute echelon inventory
        total_inventory_per_components = [0 for _ in range(self.supply_chain_config.num_components)]
        echelon_inventory = [0 for _ in range(self.supply_chain_config.num_components)]
        # Inventory on-hand and in pipeline of own node
        for i in range(self.supply_chain_config.num_components):
            total_inventory_per_components[i] = state.inventory_per_node[i] + sum(state.inventory_in_pipeline_per_node[i])
            echelon_inventory[i] = total_inventory_per_components[i]
        # Add the inventory position of the successors to compute echelon inventory position
        for i in range(self.supply_chain_config.num_components - 1, -1, -1):
            for pred in self.supply_chain_config.predecessors[i]:
                echelon_inventory[pred] += echelon_inventory[i]

        return echelon_inventory

    def get_base_stock_levels(self):
        base_stock_levels = [0 for _ in range(self.supply_chain_config.num_components)]

        # Compute the lead time per product per component, lead time is zero if that component is not part of the product
        echelon_lead_times_per_product = self.get_echelon_lead_time_per_product()
        lead_time_demand_per_product = [[0 for _ in range(self.supply_chain_config.num_components)] for _ in range(self.supply_chain_config.num_products)]
        for i in range(self.supply_chain_config.num_products):
            for j in range(self.supply_chain_config.num_components):
                lead_time_demand_per_product[i][j] = self.supply_chain_config.avg_demand_per_product[i] * echelon_lead_times_per_product[i][j]

        # Compute the base-stock levels by summing the lead time demand per component
        for i in range(self.supply_chain_config.num_components):
            for j in range(self.supply_chain_config.num_products):
                base_stock_levels[i] += lead_time_demand_per_product[j][i]

        # Perform hedging for the base-stock policy
        for i in range(self.supply_chain_config.num_components):
            base_stock_levels[i] = math.ceil(base_stock_levels[i] * (1+self.hedging_parameter))

        return base_stock_levels

    def get_echelon_lead_time_per_product(self):
        # compute echelon lead times per product
        echelon_lead_times_component_per_product = [[0 for _ in range(self.supply_chain_config.num_components)] for _ in range(self.supply_chain_config.num_products)]
        for i in range(self.supply_chain_config.num_products):
            predecessors_nodes = []
            echelon_lead_times_component_per_product[i][self.supply_chain_config.num_sub_components + i] = self.supply_chain_config.lead_times[self.supply_chain_config.num_sub_components + i]
            for j in self.supply_chain_config.predecessors[self.supply_chain_config.num_sub_components + i]:
                predecessors_nodes.append(j)
                echelon_lead_times_component_per_product[i][j] = echelon_lead_times_component_per_product[i][self.supply_chain_config.num_sub_components + i]

            # For suppliers
            # Here we that the supply chain has no more than 3 tiers
            pred_of_pred_nodes = []
            for pred in predecessors_nodes:
                echelon_lead_times_component_per_product[i][pred] += self.supply_chain_config.lead_times[pred]
                for j in self.supply_chain_config.predecessors[pred]:
                    pred_of_pred_nodes.append(j)
                    echelon_lead_times_component_per_product[i][j] = echelon_lead_times_component_per_product[i][pred]

            for pred in pred_of_pred_nodes:
                echelon_lead_times_component_per_product[i][pred] += self.supply_chain_config.lead_times[pred]

        # Add one period to echelon lead time due to periodic review
        for i in range(self.supply_chain_config.num_products):
            for j in range(self.supply_chain_config.num_components):
                if echelon_lead_times_component_per_product[i][j] > 0:
                    echelon_lead_times_component_per_product[i][j] += 1
        return echelon_lead_times_component_per_product


# Static base-stock policy
# This policy raises the echelon inventory position to the base-stock level
# Respecting material availability and capacity constraints
class StaticBaseStockPolicyRandom:
    def __init__(self, supply_chain_config, hedging_parameter, deviations = None):
        self.supply_chain_config = supply_chain_config
        self.hedging_parameter = hedging_parameter
        self.base_stock_levels = self.get_base_stock_levels()
        if deviations is not None:
            for idx, deviation in enumerate(deviations):
                self.base_stock_levels[idx] += deviation
                
        self.divergent_components = [i for i in range(self.supply_chain_config.num_components) if
                                     len(self.supply_chain_config.successors[i]) > 1]

    def set_action(self, state):
        action_per_node = [0 for _ in range(self.supply_chain_config.num_components)]
        # Randomly order the components for production decision
        indices = list(range(self.supply_chain_config.num_components))
        random.shuffle(indices)

        # Compute echelon inventory
        echelon_inventory = self.compute_echelon_inventory_position(state)

        for i in indices:
            # Raise the echelon inventory to its base-stock level if possible
            action_per_node[i] += self.base_stock_levels[i] - echelon_inventory[i]
            # Check if capacity group is not exceeded
            if (sum([action_per_node[j] for j in self.supply_chain_config.capacity_groups_indices[
                self.supply_chain_config.capacity_groups[i]]])
                    > self.supply_chain_config.capacities[self.supply_chain_config.capacity_groups[i]]):
                action_per_node[i] -= (
                        sum([action_per_node[j] for j in self.supply_chain_config.capacity_groups_indices[
                            self.supply_chain_config.capacity_groups[i]]]) - self.supply_chain_config.capacities[
                            self.supply_chain_config.capacity_groups[i]])

            # Check if material availability is not exceeded
            for j in self.supply_chain_config.predecessors[i]:
                if (state.inventory_per_node[j]
                        < sum([action_per_node[k] for k in self.supply_chain_config.successors[j]])):
                    action_per_node[i] -= (sum([action_per_node[k] for k in self.supply_chain_config.successors[j]])
                                           - state.inventory_per_node[j])

        return action_per_node

    def compute_echelon_inventory_position(self, state):
        # Compute echelon inventory
        total_inventory_per_components = [0 for _ in range(self.supply_chain_config.num_components)]
        echelon_inventory = [0 for _ in range(self.supply_chain_config.num_components)]
        for i in range(self.supply_chain_config.num_components):
            total_inventory_per_components[i] = state.inventory_per_node[i] + sum(
                state.inventory_in_pipeline_per_node[i])
            echelon_inventory[i] = total_inventory_per_components[i]
        for i in range(self.supply_chain_config.num_components - 1, -1, -1):
            for pred in self.supply_chain_config.predecessors[i]:
                echelon_inventory[pred] += echelon_inventory[i]

        return echelon_inventory

    def get_base_stock_levels(self):
        base_stock_levels = [0 for _ in range(self.supply_chain_config.num_components)]

        echelon_lead_times_per_product = self.get_echelon_lead_time_per_product()
        lead_time_demand_per_product = [[0 for _ in range(self.supply_chain_config.num_components)] for _ in
                                        range(self.supply_chain_config.num_products)]
        for i in range(self.supply_chain_config.num_products):
            for j in range(self.supply_chain_config.num_components):
                lead_time_demand_per_product[i][j] = self.supply_chain_config.avg_demand_per_product[i] * \
                                                     echelon_lead_times_per_product[i][j]

        for i in range(self.supply_chain_config.num_components):
            for j in range(self.supply_chain_config.num_products):
                base_stock_levels[i] += lead_time_demand_per_product[j][i]

        # Perform hedging for the base-stock policy
        for i in range(self.supply_chain_config.num_components):
            base_stock_levels[i] = math.ceil(base_stock_levels[i] * (1 + self.hedging_parameter))

        return base_stock_levels

    def get_echelon_lead_time_per_product(self):
        # compute echelon lead times per product
        echelon_lead_times_component_per_product = [[0 for _ in range(self.supply_chain_config.num_components)] for _ in
                                                    range(self.supply_chain_config.num_products)]
        for i in range(self.supply_chain_config.num_products):
            predecessors_nodes = []
            echelon_lead_times_component_per_product[i][self.supply_chain_config.num_sub_components + i] = (
                self.supply_chain_config.lead_times)[self.supply_chain_config.num_sub_components + i]
            for j in self.supply_chain_config.predecessors[self.supply_chain_config.num_sub_components + i]:
                predecessors_nodes.append(j)
                echelon_lead_times_component_per_product[i][j] = echelon_lead_times_component_per_product[i][
                    self.supply_chain_config.num_sub_components + i]

            # For suppliers
            # Here we that the supply chain has no more than 3 tiers
            pred_of_pred_nodes = []
            for pred in predecessors_nodes:
                echelon_lead_times_component_per_product[i][pred] += self.supply_chain_config.lead_times[pred]
                for j in self.supply_chain_config.predecessors[pred]:
                    pred_of_pred_nodes.append(j)
                    echelon_lead_times_component_per_product[i][j] = echelon_lead_times_component_per_product[i][pred]

            for pred in pred_of_pred_nodes:
                echelon_lead_times_component_per_product[i][pred] += self.supply_chain_config.lead_times[pred]

        # Add one period to echelon lead time due to periodic review
        for i in range(self.supply_chain_config.num_products):
            for j in range(self.supply_chain_config.num_components):
                if echelon_lead_times_component_per_product[i][j] > 0:
                    echelon_lead_times_component_per_product[i][j] += 1

        return echelon_lead_times_component_per_product
