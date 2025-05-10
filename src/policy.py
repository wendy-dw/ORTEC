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



# Policy Lieve
## Shortened dynamic policy, which hopefully runs faster
import random
import math

class DynamicPolicy:
    def __init__(self, config, h=0.1, shortage_history_length=20):
        self.config = config
        self.h = h
        self.shortage_history_length = shortage_history_length
        self.comp_to_node = self._map_comp_to_node()
        self.num_nodes = len(config.capacities)
        self.node_to_comps = self._map_node_to_comps()
        self.shortage_counts = {i: 0 for i in range(self.num_nodes)}
        self.comp_h_params = [h] * config.num_components
        self.shortage_history = [([False] * config.num_components) for _ in range(shortage_history_length)]
        self.history_index = 0
        self.node_importance = [0.0] * self.num_nodes
        self.demand_variability = getattr(config, "demand_variability_per_comp", [0.1] * config.num_components)
        self.base_stock_levels = self._get_base_stock_levels()

    def _map_comp_to_node(self):
        return {c: n for n, comps in enumerate(self.config.capacity_groups_indices) for c in comps}

    def _map_node_to_comps(self):
        mapping = [[] for _ in range(self.num_nodes)]
        for c, n in self.comp_to_node.items():
            mapping[n].append(c)
        return mapping

    def _calc_node_importance(self):
        node_deps = [0] * self.num_nodes
        for n in range(self.num_nodes):
            for c in self.node_to_comps[n]:
                for succ_c in self.config.successors[c]:
                    if self.comp_to_node[succ_c] != n:
                        node_deps[n] += 1
        max_deps = max(node_deps) if node_deps else 1
        self.node_importance = [(deps / max_deps) + 0.1 for deps in node_deps]

    def _get_echelon_lead_times(self):
        L = [[0 for _ in range(self.config.num_components)] for _ in range(self.config.num_products)]
        for i in range(self.config.num_products):
            root = self.config.num_sub_components + i
            L[i][root] = self.config.lead_times[root]
            for pred in self.config.predecessors[root]:
                L[i][pred] = L[i][root] + self.config.lead_times[pred]
                for pred2 in self.config.predecessors[pred]:
                    L[i][pred2] = L[i][pred] + self.config.lead_times[pred2]
        return [[lt + 1 if lt > 0 else lt for lt in prod_lts] for prod_lts in L]

    def _get_base_stock_levels(self):
        echelon_lead_times = self._get_echelon_lead_times()
        levels = [0] * self.config.num_components
        self._calc_node_importance()
        for i in range(self.config.num_products):
            for j in range(self.config.num_components):
                demand = self.config.avg_demand_per_product[i]
                lt = echelon_lead_times[i][j]
                node_importance = self.node_importance[self.comp_to_node[j]]
                levels[j] += demand * (lt + 1) * (1 + (node_importance - 0.1) * 3)
        for node_index, shortage_count in self.shortage_counts.items():
            if shortage_count > 5:
                for j in self.node_to_comps[node_index]:
                    levels[j] *= 1.2
        return [math.ceil(level * (1 + self.h)) for level in levels]

    def _compute_echelon_inventory(self, state):
        total_inv = [state.inventory_per_node[i] + sum(state.inventory_in_pipeline_per_node[i]) for i in range(self.config.num_components)]
        echelon_inv = total_inv[:]
        for i in reversed(range(self.config.num_components)):
            for pred in self.config.predecessors[i]:
                echelon_inv[pred] += echelon_inv[i]
        return echelon_inv

    def _forecast_demand(self, comp_idx, state):
        # Without historical data, we rely on the average demand from the configuration.
        return self.config.avg_demand_per_comp[comp_idx]

    def _compute_dynamic_hedging(self, comp_idx, echelon_inv, state):
        demand_var = self.demand_variability[comp_idx]
        shortage = self.base_stock_levels[comp_idx] - echelon_inv[comp_idx]
        importance = self.node_importance[comp_idx]
        dynamic_h = 1 + (self.h * demand_var * 0.5) * importance
        if shortage > 0:
            dynamic_h += self.h * (shortage / self.base_stock_levels[comp_idx]) * 0.5 * importance
        return min(dynamic_h, 1.5)

    def set_hedging_parameter(self, state):
        volatility = sum(self.demand_variability) / len(self.demand_variability) + sum(1 for i in range(self.config.num_components) if state.inventory_per_node[i] < self.base_stock_levels[i]) * 0.1
        self.h = min(0.4 + 0.2 * volatility, 0.7)

    def _is_material_available(self, state, comp_idx, amount):
        for pred_idx in self.config.predecessors[comp_idx]:
            bom_rows = self.config.bill_of_materials[(self.config.bill_of_materials.iloc[:, 0] == comp_idx) & (self.config.bill_of_materials.iloc[:, 1] == pred_idx)]
            if not bom_rows.empty and state.inventory_per_node[pred_idx] < amount * bom_rows.iloc[0, 2]:
                return False
        return True

    def _is_production_feasible(self, state, action, node_index, comp_index, amount, current_production):
        if not self._is_material_available(state, comp_index, amount):
            return False
        capacity = self.config.capacities[node_index]
        planned_production = current_production.get(node_index, 0) + amount
        return planned_production <= capacity * 1.05

    def set_action(self, state):
        echelon_inventory = self._compute_echelon_inventory(state)
        base_stock_levels = self._get_base_stock_levels()
        self.set_hedging_parameter(state)

        potential_action = [math.ceil(self.config.avg_demand_per_comp[i] * (1 + self.comp_h_params[i])) for i in range(self.config.num_components)]
        num_sub_components_bom = self.config.bill_of_materials.shape[0]
        current_shortages = [False] * self.config.num_components

        for pred_index in range(self.config.num_sub_components): # Iterate only through subcomponents
            total_needed = 0
            predecessor_name = self.config.bill_of_materials.index[pred_index]
            consuming_components = []
            for component_index in range(self.config.num_components):
                component_name = f"C{component_index + 1}"
                if (component_name in self.config.bill_of_materials.columns and
                        self.config.bill_of_materials.loc[predecessor_name, component_name] == 1 and
                        potential_action[component_index] > 0):
                    total_needed += potential_action[component_index]
                    consuming_components.append(component_index)

            if total_needed > state.inventory_per_node[pred_index]:
                shortage = total_needed - state.inventory_per_node[pred_index]
                predecessor_name = self.config.bill_of_materials.index[pred_index]
                components_to_reduce = [c_idx for c_idx in consuming_components
                                         if f"C{c_idx + 1}" in self.config.bill_of_materials.columns and
                                         self.config.bill_of_materials.loc[predecessor_name, f"C{c_idx + 1}"] == 1 and
                                         potential_action[c_idx] > 0]
                product_priorities_shortage = {
                    c_idx: (self.config.p[c_idx - self.config.num_sub_components] * (max(0, -state.inventory_per_node[c_idx]) + 1)) +
                           (base_stock_levels[c_idx] - echelon_inventory[c_idx]) * 0.3
                    if c_idx >= self.config.num_sub_components
                    else 0 for c_idx in components_to_reduce
                }
                sorted_components_to_reduce = sorted(components_to_reduce, key=lambda c_idx: product_priorities_shortage.get(c_idx, 0), reverse=False)
                reduction_needed = shortage
                for comp_idx_to_reduce in sorted_components_to_reduce:
                    reduce_by = min(potential_action[comp_idx_to_reduce], reduction_needed)
                    potential_action[comp_idx_to_reduce] = max(0, potential_action[comp_idx_to_reduce] - reduce_by)
                    reduction_needed -= reduce_by
                    if reduction_needed <= 0:
                        break

        self.shortage_history[self.history_index] = current_shortages
        self.history_index = (self.history_index + 1) % self.shortage_history_length

        max_h = 2.0
        shortage_increase_threshold = 0.2
        shortage_decrease_threshold = 0.05
        shortage_increase_factor = 0.01
        shortage_decrease_factor = 0.002
        shortage_count_decay = 0.95

        for i in range(self.config.num_components):
            shortage_freq = sum(history[i] for history in self.shortage_history) / self.shortage_history_length
            node_index = self.comp_to_node[i]
            if shortage_freq > shortage_increase_threshold and state.inventory_per_node[i] < base_stock_levels[i] * 1.1:
                self.comp_h_params[i] += shortage_increase_factor * (shortage_freq - shortage_increase_threshold)
                self.shortage_counts[node_index] += 1
            elif shortage_freq < shortage_decrease_threshold and self.comp_h_params[i] > 0.01:
                self.comp_h_params[i] -= shortage_decrease_factor
            self.shortage_counts[node_index] *= shortage_count_decay
            self.comp_h_params[i] = min(self.comp_h_params[i], max_h)
            self.comp_h_params[i] = max(self.comp_h_params[i], 0.0)

        capacity_group_priority = sorted(self.shortage_counts.items(), key=lambda item: item[1], reverse=True)
        component_order = [c for group_index, _ in capacity_group_priority for c in self.config.capacity_groups_indices[group_index]] + [i for i in range(self.config.num_components) if i not in [c for group_index, _ in capacity_group_priority for c in self.config.capacity_groups_indices[group_index]]]

        product_echelon_shortfalls = {self.config.num_sub_components + i: base_stock_levels[self.config.num_sub_components + i] - echelon_inventory[self.config.num_sub_components + i] for i in range(self.config.num_products)}

        weighted_potential_action = [0] * self.config.num_components
        for i in component_order:
            base_demand = math.ceil(self.config.avg_demand_per_comp[i] * (1 + self.h))
            weight = 1.0
            for product_index in range(self.config.num_products):
                final_product_index = self.config.num_sub_components + product_index
                component_name = f"C{i + 1}"
                product_name = self.config.bill_of_materials.index[final_product_index]
                if (component_name in self.config.bill_of_materials.columns and
                        self.config.bill_of_materials.loc[product_name, component_name] == 1):
                    weight += (product_echelon_shortfalls.get(final_product_index, 0) / (base_stock_levels[final_product_index] + 1e-9)) * 0.05
            weighted_potential_action[i] = math.ceil(base_demand * weight)
            potential_action[i] = weighted_potential_action[i]

        action_per_node = list(potential_action)

        # Enforce Capacity Constraints
        for i in range(self.config.num_nodes):
            group_indices = self.config.capacity_groups_indices[i]
            group_production = sum([action_per_node[j] for j in group_indices])
            capacity = self.config.capacities[i]
            if group_production > capacity:
                reduction_needed = group_production - capacity
                # Reduce production proportionally across the components in the group
                total_potential = sum([action_per_node[j] for j in group_indices])
                if total_potential > 0:
                    reduction_factors = [action_per_node[j] / total_potential for j in group_indices]
                    for index, comp_index in enumerate(group_indices):
                        reduction = math.ceil(reduction_needed * reduction_factors[index])
                        action_per_node[comp_index] = max(0, action_per_node[comp_index] - reduction)
                        reduction_needed -= reduction
                        if reduction_needed <= 0:
                            break
                else:
                    # If no production is planned, no reduction needed
                    pass
                self.shortage_counts[i] += 1 # Keep track of capacity issues

        # Enforce Material Availability Constraints
        for i in range(self.config.num_sub_components):
            successors = self.config.successors[i]
            needed_by_successors = sum([action_per_node[j] for j in successors])
            available_inventory = state.inventory_per_node[i]
            if needed_by_successors > available_inventory:
                reduction_needed = needed_by_successors - available_inventory
                # Reduce production of successors proportionally
                total_successor_action = sum([action_per_node[j] for j in successors])
                if total_successor_action > 0:
                    reduction_factors = [action_per_node[j] / total_successor_action for j in successors]
                    for index, successor_index in enumerate(successors):
                        reduction = math.ceil(reduction_needed * reduction_factors[index])
                        action_per_node[successor_index] = max(0, action_per_node[successor_index] - reduction)
                        reduction_needed -= reduction
                        if reduction_needed <= 0:
                            break
                else:
                    pass # No successors are planning production

        return action_per_node



class StaticPolicy:
    """
    Implements a condensed policy for inventory management with static hedging
    and without using historical demand data for hedging.
    """
    def __init__(self, config, h=0.1, shortage_history_length=20):
        self.config = config
        self.h = h  # Fixed hedging parameter
        self.shortage_history_length = shortage_history_length
        self.comp_to_node = self._map_comp_to_node()
        self.num_nodes = len(config.capacities)
        self.node_to_comps = self._map_node_to_comps()
        self.shortage_counts = {i: 0 for i in range(self.num_nodes)}
        self.comp_h_params = [h] * config.num_components  # Initialize with the fixed h
        self.shortage_history = [([False] * config.num_components) for _ in range(shortage_history_length)]
        self.history_index = 0
        self.node_importance = [0.0] * self.num_nodes
        self.demand_variability = getattr(config, "demand_variability_per_comp", [0.1] * config.num_components)
        self.base_stock_levels = self._get_base_stock_levels()

    def _map_comp_to_node(self):
        return {c: n for n, comps in enumerate(self.config.capacity_groups_indices) for c in comps}

    def _map_node_to_comps(self):
        mapping = [[] for _ in range(self.num_nodes)]
        for c, n in self.comp_to_node.items():
            mapping[n].append(c)
        return mapping

    def _calc_node_importance(self):
        node_deps = [0] * self.num_nodes
        for n in range(self.num_nodes):
            for c in self.node_to_comps[n]:
                for succ_c in self.config.successors[c]:
                    if self.comp_to_node[succ_c] != n:
                        node_deps[n] += 1
        max_deps = max(node_deps) if node_deps else 1
        self.node_importance = [(deps / max_deps) + 0.1 for deps in node_deps]

    def _get_echelon_lead_times(self):
        L = [[0 for _ in range(self.config.num_components)] for _ in range(self.config.num_products)]
        for i in range(self.config.num_products):
            root = self.config.num_sub_components + i
            L[i][root] = self.config.lead_times[root]
            for pred in self.config.predecessors[root]:
                L[i][pred] = L[i][root] + self.config.lead_times[pred]
                for pred2 in self.config.predecessors[pred]:
                    L[i][pred2] = L[i][pred] + self.config.lead_times[pred2]
        return [[lt + 1 if lt > 0 else lt for lt in prod_lts] for prod_lts in L]

    def _get_base_stock_levels(self):
        echelon_lead_times = self._get_echelon_lead_times()
        levels = [0] * self.config.num_components
        self._calc_node_importance()
        for i in range(self.config.num_products):
            for j in range(self.config.num_components):
                demand = self.config.avg_demand_per_product[i]
                lt = echelon_lead_times[i][j]
                node_importance = self.node_importance[self.comp_to_node[j]]
                levels[j] += demand * (lt + 1) * (1 + (node_importance - 0.1) * 3)
        for node_index, shortage_count in self.shortage_counts.items():
            if shortage_count > 5:
                for j in self.node_to_comps[node_index]:
                    levels[j] *= 1.2
        return [math.ceil(level * (1 + self.h)) for level in levels]

    def _compute_echelon_inventory(self, state):
        total_inv = [state.inventory_per_node[i] + sum(state.inventory_in_pipeline_per_node[i]) for i in range(self.config.num_components)]
        echelon_inv = total_inv[:]
        for i in reversed(range(self.config.num_components)):
            for pred in self.config.predecessors[i]:
                echelon_inv[pred] += echelon_inv[i]
        return echelon_inv

    def _forecast_demand(self, comp_idx, state):
        # Without historical data, we rely on the average demand from the configuration.
        return self.config.avg_demand_per_comp[comp_idx]

    def _is_material_available(self, state, comp_idx, amount):
        for pred_idx in self.config.predecessors[comp_idx]:
            bom_rows = self.config.bill_of_materials[(self.config.bill_of_materials.iloc[:, 0] == comp_idx) & (self.config.bill_of_materials.iloc[:, 1] == pred_idx)]
            if not bom_rows.empty and state.inventory_per_node[pred_idx] < amount * bom_rows.iloc[0, 2]:
                return False
        return True

    def _is_production_feasible(self, state, action, node_index, comp_index, amount, current_production):
        if not self._is_material_available(state, comp_index, amount):
            return False
        capacity = self.config.capacities[node_index]
        planned_production = current_production.get(node_index, 0) + amount
        return planned_production <= capacity * 1.05

    def set_action(self, state):
        echelon_inventory = self._compute_echelon_inventory(state)
        base_stock_levels = self._get_base_stock_levels()

        potential_action = [math.ceil(self.config.avg_demand_per_comp[i] * (1 + self.comp_h_params[i])) for i in range(self.config.num_components)]
        num_sub_components_bom = self.config.bill_of_materials.shape[0]
        current_shortages = [False] * self.config.num_components

        for pred_index in range(self.config.num_sub_components): # Iterate only through subcomponents
            total_needed = 0
            predecessor_name = self.config.bill_of_materials.index[pred_index]
            consuming_components = []
            for component_index in range(self.config.num_components):
                component_name = f"C{component_index + 1}"
                if (component_name in self.config.bill_of_materials.columns and
                        self.config.bill_of_materials.loc[predecessor_name, component_name] == 1 and
                        potential_action[component_index] > 0):
                    total_needed += potential_action[component_index]
                    consuming_components.append(component_index)

            if total_needed > state.inventory_per_node[pred_index]:
                shortage = total_needed - state.inventory_per_node[pred_index]
                predecessor_name = self.config.bill_of_materials.index[pred_index]
                components_to_reduce = [c_idx for c_idx in consuming_components
                                         if f"C{c_idx + 1}" in self.config.bill_of_materials.columns and
                                         self.config.bill_of_materials.loc[predecessor_name, f"C{c_idx + 1}"] == 1 and
                                         potential_action[c_idx] > 0]
                product_priorities_shortage = {
                    c_idx: (self.config.p[c_idx - self.config.num_sub_components] * (max(0, -state.inventory_per_node[c_idx]) + 1)) +
                           (base_stock_levels[c_idx] - echelon_inventory[c_idx]) * 0.3
                    if c_idx >= self.config.num_sub_components
                    else 0 for c_idx in components_to_reduce
                }
                sorted_components_to_reduce = sorted(components_to_reduce, key=lambda c_idx: product_priorities_shortage.get(c_idx, 0), reverse=False)
                reduction_needed = shortage
                for comp_idx_to_reduce in sorted_components_to_reduce:
                    reduce_by = min(potential_action[comp_idx_to_reduce], reduction_needed)
                    potential_action[comp_idx_to_reduce] = max(0, potential_action[comp_idx_to_reduce] - reduce_by)
                    reduction_needed -= reduce_by
                    if reduction_needed <= 0:
                        break

        self.shortage_history[self.history_index] = current_shortages
        self.history_index = (self.history_index + 1) % self.shortage_history_length

        capacity_group_priority = sorted(self.shortage_counts.items(), key=lambda item: item[1], reverse=True)
        component_order = [c for group_index, _ in capacity_group_priority for c in self.config.capacity_groups_indices[group_index]] + [i for i in range(self.config.num_components) if i not in [c for group_index, _ in capacity_group_priority for c in self.config.capacity_groups_indices[group_index]]]

        product_echelon_shortfalls = {self.config.num_sub_components + i: base_stock_levels[self.config.num_sub_components + i] - echelon_inventory[self.config.num_sub_components + i] for i in range(self.config.num_products)}

        weighted_potential_action = [0] * self.config.num_components
        for i in component_order:
            base_demand = math.ceil(self.config.avg_demand_per_comp[i] * (1 + self.h))
            weight = 1.0
            for product_index in range(self.config.num_products):
                final_product_index = self.config.num_sub_components + product_index
                component_name = f"C{i + 1}"
                product_name = self.config.bill_of_materials.index[final_product_index]
                if (component_name in self.config.bill_of_materials.columns and
                        self.config.bill_of_materials.loc[product_name, component_name] == 1):
                    weight += (product_echelon_shortfalls.get(final_product_index, 0) / (base_stock_levels[final_product_index] + 1e-9)) * 0.05
            weighted_potential_action[i] = math.ceil(base_demand * weight)
            potential_action[i] = weighted_potential_action[i]

        action_per_node = list(potential_action)

        # Enforce Capacity Constraints
        for i in range(self.config.num_nodes):
            group_indices = self.config.capacity_groups_indices[i]
            group_production = sum([action_per_node[j] for j in group_indices])
            capacity = self.config.capacities[i]
            if group_production > capacity:
                reduction_needed = group_production - capacity
                # Reduce production proportionally across the components in the group
                total_potential = sum([action_per_node[j] for j in group_indices])
                if total_potential > 0:
                    reduction_factors = [action_per_node[j] / total_potential for j in group_indices]
                    for index, comp_index in enumerate(group_indices):
                        reduction = math.ceil(reduction_needed * reduction_factors[index])
                        action_per_node[comp_index] = max(0, action_per_node[comp_index] - reduction)
                        reduction_needed -= reduction
                        if reduction_needed <= 0:
                            break
                else:
                    # If no production is planned, no reduction needed
                    pass
                self.shortage_counts[i] += 1 # Keep track of capacity issues

        # Enforce Material Availability Constraints
        for i in range(self.config.num_sub_components):
            successors = self.config.successors[i]
            needed_by_successors = sum([action_per_node[j] for j in successors])
            available_inventory = state.inventory_per_node[i]
            if needed_by_successors > available_inventory:
                reduction_needed = needed_by_successors - available_inventory
                # Reduce production of successors proportionally
                total_successor_action = sum([action_per_node[j] for j in successors])
                if total_successor_action > 0:
                    reduction_factors = [action_per_node[j] / total_successor_action for j in successors]
                    for index, successor_index in enumerate(successors):
                        reduction = math.ceil(reduction_needed * reduction_factors[index])
                        action_per_node[successor_index] = max(0, action_per_node[successor_index] - reduction)
                        reduction_needed -= reduction
                        if reduction_needed <= 0:
                            break
                else:
                    pass # No successors are planning production

        return action_per_node
