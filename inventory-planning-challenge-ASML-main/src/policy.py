import random
import math
import numpy as np
from scipy.stats import norm

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


import math
import random

class EchelonBaseStockPolicy:
    def __init__(self, supply_chain_config, hedging_parameter):
        self.sc = supply_chain_config
        self.hedge = hedging_parameter
        # Precompute static S_i targets
        self.S = self._compute_targets()

    def _compute_targets(self):
        """
        For each component i, sum mean downstream demand over the full
        echelon lead time, then apply hedging.
        """
        # 1. Compute echelon lead times for each component:
        #    same structure as your existing helper, but per component
        L = [0]*self.sc.num_components
        # For each final product j, propagate lead times upstream
        for j in range(self.sc.num_products):
            prod_idx = self.sc.num_sub_components + j
            # start with its own lead time
            queue = [(prod_idx, self.sc.lead_times[prod_idx])]
            visited = set()
            while queue:
                node, lead = queue.pop(0)
                if node in visited: continue
                visited.add(node)
                L[node] = lead
                for pred in self.sc.predecessors[node]:
                    # upstream lead = current lead + its own lead time
                    queue.append((pred, lead + self.sc.lead_times[pred]))
        # 2. For each component i, sum avg demand of each product * L_i
        base = [0]*self.sc.num_components
        for i in range(self.sc.num_components):
            for j in range(self.sc.num_products):
                base[i] += self.sc.avg_demand_per_product[j] * L[i]
        # 3. Apply hedging and ceil
        return [math.ceil(base[i] * (1 + self.hedge)) for i in range(self.sc.num_components)]

    def set_action(self, state):
        # Step 1: compute echelon inventory positions
        echelon = [0]*self.sc.num_components
        # on‐hand + in‐pipeline
        for i in range(self.sc.num_components):
            echelon[i] = state.inventory_per_node[i] + sum(state.inventory_in_pipeline_per_node[i])
        # add downstream
        for i in range(self.sc.num_components-1, -1, -1):
            for pred in self.sc.predecessors[i]:
                echelon[pred] += echelon[i]

        # Step 2: order‑up‑to S_i
        action = [max(0, self.S[i] - echelon[i]) for i in range(self.sc.num_components)]

        # Step 3: enforce capacity‐group constraints
        for node_group_idx, group in enumerate(self.sc.capacity_groups_indices):
            total = sum(action[i] for i in group)
            cap = self.sc.capacities[node_group_idx]
            if total > cap:
                # scale down proportionally
                factor = cap / total
                for i in group:
                    action[i] = math.floor(action[i] * factor)

        # Step 4: enforce material‐availability constraints
        # If a predecessor j doesn’t have enough on‑hand to feed successors,
        # reduce each successor proportionally.
        for j in range(self.sc.num_sub_components):
            required = sum(action[k] for k in self.sc.successors[j])
            avail = state.inventory_per_node[j]
            if required > avail and required > 0:
                factor = avail / required
                for k in self.sc.successors[j]:
                    action[k] = math.floor(action[k] * factor)

        return action

#  Non‐Stationary Base‐Stock: time‐varying S_t from ARMA forecast
class NonstationaryBaseStockPolicy:
    def __init__(self, supply_chain_config, service_level):
        self.sc = supply_chain_config
        # z‐score for desired service level
        self.z = norm.ppf(service_level)

    def set_action(self, state):
        # 1. Forecast aggregate demand for next L_i periods via ARMA
        ar_params = np.array(self.sc.aggregate_demand_ar_params)
        history  = np.array(state.aggregate_demand_model_history)
        μ        = self.sc.aggregate_demand_mean
        σ        = self.sc.aggregate_demand_error_sigma

        # Compute echelon lead times per component (reuse logic)
        L = [0]*self.sc.num_components
        for j in range(self.sc.num_products):
            idx = self.sc.num_sub_components + j
            queue = [(idx, self.sc.lead_times[idx])]
            seen  = set()
            while queue:
                node, lead = queue.pop(0)
                if node in seen: continue
                seen.add(node)
                L[node] = lead
                for p in self.sc.predecessors[node]:
                    queue.append((p, lead + self.sc.lead_times[p]))

        # 2. Compute S_t[i] = Σ_{k=1..L[i]} E[d_{t+k}] + z*√(Σσ²)
        #    Here E[d_{t+k}] from AR model; error var ~ σ² each step
        #    For simplicity, use μ for each future, and σ√L
        S_t = [0]*self.sc.num_components
        for i in range(self.sc.num_components):
            lead = L[i]
            forecast_sum = lead * μ
            safety_stock = self.z * σ * math.sqrt(lead)
            S_t[i] = math.ceil((forecast_sum + safety_stock) * 
                               (1 + 0))  # no extra hedge here

        # 3. Compute echelon inventory
        echelon = [0]*self.sc.num_components
        for i in range(self.sc.num_components):
            echelon[i] = state.inventory_per_node[i] + sum(state.inventory_in_pipeline_per_node[i])
        for i in range(self.sc.num_components-1, -1, -1):
            for p in self.sc.predecessors[i]:
                echelon[p] += echelon[i]

        # 4. Order‐up‐to S_t
        action = [max(0, S_t[i] - echelon[i]) for i in range(self.sc.num_components)]

        # 5. Enforce capacity & material feasibility (same as others)
        #    Capacity
        for gid, group in enumerate(self.sc.capacity_groups_indices):
            total = sum(action[i] for i in group)
            cap   = self.sc.capacities[gid]
            if total > cap:
                factor = cap/total
                for i in group:
                    action[i] = math.floor(action[i]*factor)
        #    Material
        for j in range(self.sc.num_sub_components):
            req = sum(action[k] for k in self.sc.successors[j])
            avail = state.inventory_per_node[j]
            if req>avail and req>0:
                f = avail/req
                for k in self.sc.successors[j]:
                    action[k] = math.floor(action[k]*f)

        return action


# (s, S) Reorder‐Point Policy
class sSPolicy:
    def __init__(self, supply_chain_config, s_levels, S_levels):
        self.sc = supply_chain_config
        self.s  = s_levels
        self.S  = S_levels

    def set_action(self, state):
        action = [0]*self.sc.num_components
        # inventory position = on‐hand + pipeline
        ip = [state.inventory_per_node[i] + sum(state.inventory_in_pipeline_per_node[i])
              for i in range(self.sc.num_components)]
        for i in range(self.sc.num_components):
            if ip[i] < self.s[i]:
                action[i] = max(0, self.S[i] - ip[i])
        # capacity
        for gid, group in enumerate(self.sc.capacity_groups_indices):
            tot = sum(action[i] for i in group)
            if tot > self.sc.capacities[gid]:
                factor = self.sc.capacities[gid]/tot
                for i in group:
                    action[i] = math.floor(action[i]*factor)
        # material
        for j in range(self.sc.num_sub_components):
            req = sum(action[k] for k in self.sc.successors[j])
            avail = state.inventory_per_node[j]
            if req>avail and req>0:
                f = avail/req
                for k in self.sc.successors[j]:
                    action[k] = math.floor(action[k]*f)
        return action


#Dual‐Index Policy (lower = on‐hand, upper = on‐hand+pipeline)
class DualIndexPolicy:
    def __init__(self, supply_chain_config, s_lower, S_upper):
        self.sc = supply_chain_config
        self.s = s_lower
        self.S = S_upper

    def set_action(self, state):
        action = [0]*self.sc.num_components
        for i in range(self.sc.num_components):
            on_hand = state.inventory_per_node[i]
            upper   = on_hand + sum(state.inventory_in_pipeline_per_node[i])
            if on_hand < self.s[i]:
                action[i] = max(0, self.S[i] - upper)
        # capacity
        for gid, group in enumerate(self.sc.capacity_groups_indices):
            tot = sum(action[i] for i in group)
            if tot > self.sc.capacities[gid]:
                f = self.sc.capacities[gid]/tot
                for i in group:
                    action[i] = math.floor(action[i]*f)
        # material
        for j in range(self.sc.num_sub_components):
            req   = sum(action[k] for k in self.sc.successors[j])
            avail = state.inventory_per_node[j]
            if req>avail and req>0:
                f = avail/req
                for k in self.sc.successors[j]:
                    action[k] = math.floor(action[k]*f)
        return action