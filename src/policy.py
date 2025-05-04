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
class MySmartPolicy:
    """
    Implements a smart policy for inventory management in a multi-echelon supply chain.

    This policy calculates production actions based on base stock levels,
    demand forecasting, and dynamic hedging, while considering capacity constraints
    and material availability. It also incorporates a mechanism to adjust hedging
    parameters based on shortage history.  It now includes node importance.
    """

    def __init__(self, config, hedging_parameter=0.1, shortage_history_length=20):
        """
        Initializes the MySmartPolicy object.
    
        Args:
            config: Configuration object containing supply chain parameters.
            hedging_parameter: Initial hedging parameter.
            shortage_history_length: Length of the shortage history to track.
        """
        self.config = config
        self.hedging_parameter = hedging_parameter
        self.shortage_history_length = shortage_history_length
    
        # Pre-compute and store mappings for efficiency
        self.component_to_node = self._map_component_to_node()
        self.node_to_components = self._map_node_to_components()  # New mapping
    
        # Initialize dynamic attributes
        self.num_nodes = self.config.num_components
        self.current_node = 0
        self.bottleneck_counts = {i: 0 for i in range(len(self.config.capacities))}
        self.component_shortage_counts = [0] * self.config.num_components
        self.component_hedging_parameters = [hedging_parameter] * self.config.num_components
        self.shortage_history = [([False] * self.config.num_components) for _ in range(shortage_history_length)]
        self.history_index = 0
    
        self.component_importance = [1.0] * self.config.num_components  # Default importance
        self.node_shortage_counts = {i: 0 for i in range(len(self.config.capacities))}  # Track node shortages
        self.node_importance = [0.0] * len(self.config.capacities) # Initialize node importance
        self.demand_variability_per_comp = (
            self.config.demand_variability_per_comp
            if hasattr(self.config, "demand_variability_per_comp")
            else [0.1] * self.config.num_components
        )
    
        self.node_shortage_history = {i: [] for i in range(len(self.config.capacities))} # Initialize HERE
        self.node_shortage_threshold = 3 # Example threshold for triggering increased base stock
    
        self.base_stock_levels = self._get_base_stock_levels()
    
    def _map_component_to_node(self):
        """
        Creates a mapping from component index to node index.
    
        This mapping is based on the capacity groups defined in the configuration.
        """
        mapping = {}
        for node_idx, comp_indices in enumerate(self.config.capacity_groups_indices):
            for comp_index in comp_indices:
                mapping[comp_index] = node_idx
        return mapping
    
    def _map_node_to_components(self):
        """
        Creates a mapping from node index to a list of component indices.
        """
        mapping = [[] for _ in range(len(self.config.capacities))]
        for comp_index, node_index in self.component_to_node.items():
            mapping[node_index].append(comp_index)
        return mapping
    
    def _calculate_node_importance(self):
        """
        Calculates the importance of each node based on how many other nodes
        depend on it (through its components).
        """
        node_dependencies = [0] * len(self.config.capacities)
        for node_index in range(len(self.config.capacities)):
            components_at_node = self.node_to_components[node_index]
            for comp_index in components_at_node:
                for successor_comp_index in self.config.successors[comp_index]:
                    successor_node_index = self.component_to_node[successor_comp_index]
                    if successor_node_index != node_index:  # Don't count self-dependency
                        node_dependencies[node_index] += 1
    
        # Normalize node importance, and add a small constant.
        max_dependencies = max(node_dependencies) if node_dependencies else 1
        self.node_importance = [(deps / max_dependencies) + 0.1 for deps in node_dependencies] # Ensure importance is > 0
    
    def _get_base_stock_levels(self):
        """
        Calculates the base stock levels for each component.
    
        Base stock level is calculated based on echelon lead times, average demand,
        and the hedging parameter.
        """
        echelon_lead_times = self._get_echelon_lead_time_per_product()
        levels = [0] * self.config.num_components
    
        # Calculate node importance.
        self._calculate_node_importance()
    
        for i in range(self.config.num_products):
            for j in range(self.config.num_components):
                demand = self.config.avg_demand_per_product[i]
                lead_time = echelon_lead_times[i][j]
                node_index = self.component_to_node[j]
                node_importance_factor = self.node_importance[node_index]
                # Make the effect of node importance more extreme:
                levels[j] += demand * (lead_time + 1) * (1 + (node_importance_factor - 0.1) * 3)  # Increased factor
    
            # Further increase base stock levels for nodes with high shortage counts
            if hasattr(self, 'node_shortage_counts'): # Check if the attribute exists
                for node_index, shortage_count in self.node_shortage_counts.items():
                    if shortage_count > 5:  # You can adjust this threshold
                        for j in self.node_to_components[node_index]:
                            levels[j] *= 1.2  # Increase by 20% (adjust as needed)
    
        levels = [0] * self.config.num_components
        self._calculate_node_importance()
    
        for i in range(self.config.num_products):
            for j in range(self.config.num_components):
                demand = self.config.avg_demand_per_product[i]
                lead_time = echelon_lead_times[i][j]
                node_index = self.component_to_node[j]
                node_importance_factor = self.node_importance[node_index]
                levels[j] += demand * (lead_time + 1) * (1 + (node_importance_factor - 0.1) * 2)
    
        # Dynamically adjust based on node shortage history
        for node_index in range(len(self.config.capacities)):
            if sum(self.node_shortage_history[node_index][-10:]) > self.node_shortage_threshold: # Check recent shortages
                for j in self.node_to_components[node_index]:
                    levels[j] *= 1.15 # Temporarily increase base stock
    
        return [math.ceil(level * (1 + self.hedging_parameter)) for level in levels]
    
    

    def _get_echelon_lead_time_per_product(self):
        """
        Calculates the echelon lead time for each product and component.

        This method determines the longest path (lead time) from each component
        to the final product.
        """
        num_products = self.config.num_products
        num_components = self.config.num_components
        L = [[0 for _ in range(num_components)] for _ in range(num_products)]

        for i in range(num_products):
            root = self.config.num_sub_components + i
            L[i][root] = self.config.lead_times[root]

            for pred in self.config.predecessors[root]:
                L[i][pred] = L[i][root] + self.config.lead_times[pred]

                for pred2 in self.config.predecessors[pred]:
                    L[i][pred2] = L[i][pred] + self.config.lead_times[pred2]

        # Add 1 to all lead times, as in the original code.
        for i in range(num_products):
            for j in range(num_components):
                if L[i][j] > 0:
                    L[i][j] += 1
        return L

    def _compute_echelon_inventory_position(self, state):
        """
        Calculates the echelon inventory position for each component.

        Echelon inventory is the total inventory at a given stage and all downstream stages.
        """
        total_inventory = [
            state.inventory_per_node[i] + sum(state.inventory_in_pipeline_per_node[i])
            for i in range(self.config.num_components)
        ]
        echelon_inventory = total_inventory[:]  # Copy the list

        for i in reversed(range(self.config.num_components)):
            for pred in self.config.predecessors[i]:
                echelon_inventory[pred] += echelon_inventory[i]
        return echelon_inventory

    def _forecast_demand(self, component_idx, state, window_size=5):
        """
        Forecasts demand for a component using a moving average.

        Args:
            component_idx: Index of the component to forecast demand for.
            state: Current state of the supply chain.
            window_size: Number of periods to use in the moving average.

        Returns:
            The forecasted demand for the component.
        """
        historical_demand = self._get_historical_demand(component_idx, state, window_size)
        if not historical_demand:
            return self.config.avg_demand_per_comp[component_idx]
        return sum(historical_demand) / len(historical_demand)

    def _get_historical_demand(self, component_idx, state, window_size):
        """
        Retrieves historical demand data for a component.

        This is a placeholder method.  In a real application, this would
        access a data source or the state object to get actual historical demand.

        Args:
            component_idx: The index of the component.
            state: The current state.
            window_size: The number of historical periods to retrieve.

        Returns:
            A list of historical demand values.  Returns an empty list if no data.
        """
        # Placeholder: Return a list of random demand values.  Replace this.
        return [random.randint(50, 150) for _ in range(window_size)]  # Mock data

    def _compute_demand_variability(self, component_idx):
        """
        Computes the demand variability for a component.

        This method calculates the variability of demand for a given component.
        It uses the pre-defined variability from the config if available.

        Args:
            component_idx: The index of the component.

        Returns:
            The demand variability for the component.  Higher = more variable.
        """
        return self.demand_variability_per_comp[component_idx]

    def _compute_dynamic_hedging(self, component_idx, echelon_inventory, state):
        """
        Dynamically adjusts the hedging parameter for a component.

        The hedging parameter is increased if the component has high demand
        variability or is experiencing a shortage.

        Args:
            component_idx: The index of the component.
            echelon_inventory: The current echelon inventory levels.
            state: The current state of the system.

        Returns:
            The adjusted hedging parameter for the component.
        """
        demand_variability = self._compute_demand_variability(component_idx)
        component_shortage = self.base_stock_levels[component_idx] - echelon_inventory[component_idx]

        importance_factor = self.component_importance[component_idx]
        dynamic_hedging = 1 + (self.hedging_parameter * demand_variability * 0.5) * importance_factor
        if component_shortage > 0:
            dynamic_hedging += self.hedging_parameter * (component_shortage / self.base_stock_levels[component_idx]) * 0.5 * importance_factor
        return min(dynamic_hedging, 1.5)

    def set_hedging_parameter(self, state):
        """
        Adjusts the global hedging parameter based on overall system state.

        This adjustment considers demand volatility and overall inventory levels.

        Args:
            state: The current state of the supply chain.
        """
        volatility_factor = self._compute_volatility_factor(state)
        self.hedging_parameter = min(0.4 + 0.2 * volatility_factor, 0.7)  # Reduced cap and sensitivity

    def _compute_volatility_factor(self, state):
        """
        Calculates a volatility factor based on inventory and demand.

        This factor is used to adjust the hedging parameter.

        Args:
            state: The current state of the supply chain.

        Returns:
            A value representing the overall volatility in the system.
        """
        volatility = sum(self.demand_variability_per_comp) / len(self.demand_variability_per_comp)
        low_inventory = sum(
            1 for i in range(self.config.num_components)
            if state.inventory_per_node[i] < self.base_stock_levels[i]
        )
        return volatility + low_inventory * 0.1  # Simple combination

    def _is_material_available(self, state, comp_index, produce_amount):
        """
        Checks if enough materials are available to produce a given amount of a component.

        Args:
            state: The current state of the supply chain.
            comp_index: The index of the component to produce.
            produce_amount: The amount of the component to produce.

        Returns:
            True if materials are sufficient, False otherwise.
        """
        for pred_index in self.config.predecessors[comp_index]:
            bom_rows = self.config.bill_of_materials[
                (self.config.bill_of_materials.iloc[:, 0] == comp_index)
                & (self.config.bill_of_materials.iloc[:, 1] == pred_index)
            ]
            if not bom_rows.empty:
                required_quantity = bom_rows.iloc[0, 2]
                if state.inventory_per_node[pred_index] < produce_amount * required_quantity:
                    print(f"  Not enough predecessor {pred_index} for component {comp_index}")
                    return False
        return True

    def _is_production_feasible(self, state, action, node_index, comp_index, produce_amount, current_production_per_node):
        """
        Checks if producing a given amount of a component is feasible at a node.

        Considers both material availability and capacity constraints.

        Args:
            state: The current state of the supply chain.
            action: The planned production actions.  (Not used here, but kept for consistency).
            node_index: The index of the node where production is being considered.
            comp_index: The index of the component to produce.
            produce_amount: The amount of the component to produce.
            current_production_per_node: A dictionary tracking current production at each node.

        Returns:
            True if production is feasible, False otherwise.
        """
        # Check material availability
        if not self._is_material_available(state, comp_index, produce_amount):
            return False

        # Check capacity
        capacity = self.config.capacities[node_index]
        planned_production_at_node = current_production_per_node.get(node_index, 0) + produce_amount
        if planned_production_at_node > capacity * 1.05:  # 5% slack
            print(f"  Exceeds capacity at node {node_index} for component {comp_index}")
            return False
        return True

    def _is_feasible(self, state, action, node):
        """
        Checks if a given action is feasible at a node, considering predecessors and capacity.

        Args:
            state: The current state of the supply chain.
            action: The production action to check.
            node: The node index.

        Returns:
            True if the action is feasible, False otherwise.
        """
        for pred in self.config.predecessors[node]:
            total_required = sum(action[succ] for succ in self.config.successors[pred])
            if state.inventory_per_node[pred] < total_required * 0.95:  # 5% slack
                print(
                    f"Action infeasible due to predecessor {pred}: required {total_required}, "
                    f"available {state.inventory_per_node[pred]}"
                )
                return False

        cap_group = self.config.capacity_groups[node]
        cap_indices = self.config.capacity_groups_indices[cap_group]
        total_action = sum(action[j] for j in cap_indices)
        if total_action > self.config.capacities[cap_group] * 1.05:  # 5% slack
            print(
                f"Action infeasible due to capacity group {cap_group}: total action {total_action}, "
                f"capacity {self.config.capacities[cap_group]}"
            )
            return False
        return True

    def set_action(self, state):
        """
        Calculates the production actions for all components.
        """
        echelon_inventory = self._compute_echelon_inventory_position(state)
        base_stock_levels = self._get_base_stock_levels() # Get updated base stock levels
        adaptive_hedging_parameter = self.hedging_parameter

        # Update shortage history
        current_shortages = [False] * self.config.num_components
        num_sub_components_bom = self.config.bill_of_materials.shape[0]

        # Initial potential actions, using the component hedging parameters
        potential_action = [
            math.ceil(self.config.avg_demand_per_comp[i] * (1 + self.component_hedging_parameters[i]))
            for i in range(self.config.num_components)
        ]

        # Determine subcomponent demand and shortages.
        total_subcomponent_demand = {}
        for predecessor_index in range(num_sub_components_bom):
            total_needed = 0
            predecessor_name = self.config.bill_of_materials.index[predecessor_index]
            for component_index in range(self.config.num_components):
                component_name = f"C{component_index + 1}"
                if (
                    component_name in self.config.bill_of_materials.columns
                    and self.config.bill_of_materials.loc[predecessor_name, component_name] == 1
                    and potential_action[component_index] > 0
                ):
                    total_needed += potential_action[component_index]

            if total_needed > 0 and predecessor_index < self.config.num_sub_components:
                total_subcomponent_demand[predecessor_index] = total_needed
                available = state.inventory_per_node[predecessor_index]
                if total_needed > available:
                    current_shortages[predecessor_index] = True

        # Update shortage history.
        self.shortage_history[self.history_index] = current_shortages
        self.history_index = (self.history_index + 1) % self.shortage_history_length

        # Adjust component hedging parameters based on recent shortage history
        max_hedging_parameter = 2.0  # Introduce a maximum value
        shortage_threshold_increase = 0.2
        shortage_threshold_decrease = 0.05
        shortage_increase_factor = 0.01  # Smaller increase
        shortage_decrease_factor = 0.002 # Smaller decrease
        shortage_count_decay = 0.95     # Decay factor for shortage counts

        for i in range(self.config.num_components):
            shortage_frequency = sum(history[i] for history in self.shortage_history) / self.shortage_history_length
            node_index = self.component_to_node[i]

            # Adjust hedging parameter based on shortage frequency and inventory
            if shortage_frequency > shortage_threshold_increase:
                inventory_level = state.inventory_per_node[i]
                if inventory_level < base_stock_levels[i] * 1.1: # Don't increase if inventory is already high
                    self.component_hedging_parameters[i] += shortage_increase_factor * (shortage_frequency - shortage_threshold_increase)
                    self.node_shortage_counts[node_index] += 1
            elif shortage_frequency < shortage_threshold_decrease and self.component_hedging_parameters[i] > 0.01:
                self.component_hedging_parameters[i] -= shortage_decrease_factor
                self.node_shortage_counts[node_index] *= shortage_count_decay # Apply decay
            else:
                self.node_shortage_counts[node_index] *= shortage_count_decay # Apply decay even if no change

            self.component_hedging_parameters[i] = min(self.component_hedging_parameters[i], max_hedging_parameter)
            self.component_hedging_parameters[i] = max(self.component_hedging_parameters[i], 0.0) # Ensure it doesn't go negative

        # Re-calculate potential action with adjusted hedging
        potential_action = [
            math.ceil(self.config.avg_demand_per_comp[i] * (1 + self.component_hedging_parameters[i]))
            for i in range(self.config.num_components)
        ]

        # Prioritize component order based on bottleneck frequency
        capacity_group_priority = sorted(
            self.bottleneck_counts.items(), key=lambda item: item[1], reverse=True
        )
        component_order = []
        processed_components = set()
        for group_index, _ in capacity_group_priority:
            for comp_index in self.config.capacity_groups_indices[group_index]:
                if comp_index not in processed_components:
                    component_order.append(comp_index)
                    processed_components.add(comp_index)
        component_order.extend(
            i for i in range(self.config.num_components) if i not in processed_components
        )  # Add remaining

        # Prioritize initial potential actions based on echelon inventory shortfall
        product_echelon_shortfalls = {
            self.config.num_sub_components + i: base_stock_levels[self.config.num_sub_components + i]
            - echelon_inventory[self.config.num_sub_components + i]
            for i in range(self.config.num_products)
        }

        # Calculate weighted potential actions
        weighted_potential_action = [0] * self.config.num_components
        for i in component_order:
            base_demand = math.ceil(self.config.avg_demand_per_comp[i] * (1 + adaptive_hedging_parameter))
            weight = 1.0
            for product_index in range(self.config.num_products):
                final_product_index = self.config.num_sub_components + product_index
                component_name = f"C{i + 1}"
                product_name = self.config.bill_of_materials.index[final_product_index]
                if (
                    component_name in self.config.bill_of_materials.columns
                    and self.config.bill_of_materials.loc[product_name, component_name] == 1
                ):
                    weight += (
                        product_echelon_shortfalls.get(final_product_index, 0)
                        / (base_stock_levels[final_product_index] + 1e-9) # Avoid division by zero
                    ) * 0.05  # Reduced weight
            weighted_potential_action[i] = math.ceil(base_demand * weight)
            potential_action[i] = weighted_potential_action[i]  # Use weighted action

        # Adjust potential actions to handle subcomponent shortages
        total_subcomponent_demand = {}
        num_sub_components_bom = self.config.bill_of_materials.shape[0]
        for predecessor_index in range(num_sub_components_bom):
            total_needed = 0
            predecessor_name = self.config.bill_of_materials.index[predecessor_index]
            for component_index in range(self.config.num_components):
                component_name = f"C{component_index + 1}"
                if (
                    component_name in self.config.bill_of_materials.columns
                    and self.config.bill_of_materials.loc[predecessor_name, component_name] == 1
                    and potential_action[component_index] > 0
                ):
                    total_needed += potential_action[component_index]
            if total_needed > 0 and predecessor_index < self.config.num_sub_components:
                total_subcomponent_demand[predecessor_index] = total_needed

        print("\nTotal Sub-component Demand:")  # Debugging output
        for sub_comp_index, total_needed in total_subcomponent_demand.items():
            available = state.inventory_per_node[sub_comp_index]
            print(f"Component {sub_comp_index}: Needed = {total_needed}, Available = {available}")
            if total_needed > available:
                shortage = total_needed - available
                print(f"  Shortage of {shortage} for component {sub_comp_index}!")
                predecessor_name = self.config.bill_of_materials.index[sub_comp_index]
                components_to_reduce = []
                product_priorities_shortage = {}  # Priority based on shortage

                for component_index in range(self.config.num_components):
                    component_name = f"C{component_index + 1}"
                    if (
                        component_name in self.config.bill_of_materials.columns
                        and self.config.bill_of_materials.loc[predecessor_name, component_name] == 1
                        and potential_action[component_index] > 0
                    ):
                        components_to_reduce.append(component_index)
                        if component_index >= self.config.num_sub_components:  # It's a final product
                            product_index = component_index - self.config.num_sub_components
                            backlog = max(0, -state.inventory_per_node[component_index])
                            echelon_shortfall = (
                                base_stock_levels[component_index] - echelon_inventory[component_index]
                            )
                            # Priority based on echelon inventory shortfall, backlog, and penalty
                            priority = (
                                self.config.p[product_index] * (backlog + 1)
                            ) + (
                                echelon_shortfall * 0.3 # Adjust weights as needed
                            )
                            product_priorities_shortage[component_index] = priority
                        else:
                            product_priorities_shortage[component_index] = 0

                sorted_components_to_reduce = sorted(
                    components_to_reduce,
                    key=lambda comp_index: product_priorities_shortage.get(comp_index, 0),
                    reverse=False,
                )

                reduction_needed = shortage
                for comp_index_to_reduce in sorted_components_to_reduce:
                    reduce_by = min(potential_action[comp_index_to_reduce], reduction_needed)
                    potential_action[comp_index_to_reduce] = max(
                        0, potential_action[comp_index_to_reduce] - reduce_by
                    )
                    reduction_needed -= reduce_by
                    print(
                        f"  Reduced production of component {comp_index_to_reduce} to "
                        f"{potential_action[comp_index_to_reduce]}"
                    )  # Debug
                    if reduction_needed <= 0:
                        break

        # Capacity constraints check and bottleneck count update
        action_per_node = list(potential_action)  # Create a copy to avoid modifying original
        for i in range(self.config.num_components):
            capacity_group = self.config.capacity_groups[i]
            capacity = self.config.capacities[capacity_group]
            group_production = sum(
                [action_per_node[j] for j in self.config.capacity_groups_indices[capacity_group]]
            )
            if group_production > capacity:
                reduction_needed = group_production - capacity
                action_per_node[i] = max(0, action_per_node[i] - reduction_needed)
                self.bottleneck_counts[capacity_group] += 1

        print(
            f"\nSmart Action (All Nodes): {action_per_node}, Hedging: {self.component_hedging_parameters}"
        )  # Debug
        return action_per_node
