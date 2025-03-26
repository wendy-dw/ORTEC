import pandas as pd


class SupplyChainConfig:
    def __init__(self):
        path_data = 'data/supply_chain_configuration.xlsx' # Path to the data file
        self.bill_of_materials = pd.read_excel(path_data, sheet_name='BillOfMaterials')
        component_names = self.bill_of_materials.iloc[:, 0].to_list()
        self.successors, self.predecessors = self.get_bill_of_materials()
        self.lead_times = pd.read_excel(path_data,
                                        sheet_name='LeadTimes').iloc[:, 1].to_list()
        # Capacities per capacity group
        self.capacities = pd.read_excel(path_data,
                                        sheet_name='Capacities', index_col=None).iloc[:, 1].to_list()
        df_capacities_groups = pd.read_excel(path_data,
                                             sheet_name='Capacities', index_col=None).iloc[:, 2:]
        # Define the capacity groups
        self.capacity_groups_indices = self.get_capacity_groups(df_capacities_groups, component_names)

        self.num_components = len(self.lead_times)
        self.num_sub_components = sum(1 for successors in self.successors if successors)
        self.num_products = self.num_components - self.num_sub_components
        self.product_indices = [self.num_components + i for i in range(self.num_products)]
        self.num_nodes = len(self.capacity_groups_indices)

        # Find the capacity group for each component/product
        self.capacity_groups = self.find_capacity_groups()
        self.h = pd.read_excel(path_data,
                               sheet_name='HoldingCosts').iloc[:, 1].to_list()
        self.p = pd.read_excel(path_data,
                               sheet_name='BacklogPenalties').iloc[:self.num_products, 1].to_list()

        # Aggregate demand model parameters
        self.aggregate_demand_ar_params = pd.read_excel(path_data,
                                                        sheet_name='AggregateDemandModel',
                                                        header=None).iloc[0, 1:].to_list()
        self.aggregate_demand_num_lags = len(self.aggregate_demand_ar_params)
        self.aggregate_demand_error_sigma = pd.read_excel(path_data,
                                                          sheet_name='AggregateDemandModel',
                                                          header=None).iloc[1, 1]

        # Mean demand per product and split parameters
        self.product_demand_mean = pd.read_excel(path_data,
                                                 sheet_name='ProductDemandMean').iloc[:self.num_products, 1].to_list()

        self.product_demand_split_kappa = pd.read_excel(
            path_data, sheet_name='ProductDemandMean').iloc[:self.num_products, 2].to_list()
        self.product_demand_split_theta = pd.read_excel(
            path_data, sheet_name='ProductDemandMean').iloc[:self.num_products, 3].to_list()

        self.aggregate_demand_mean = sum(self.product_demand_mean)

        self.avg_demand_per_product = self.product_demand_mean

        # Average demand per component for initial state
        self.avg_demand_per_comp = self.get_avg_demand()

    def get_bill_of_materials(self):
        # Vector of vectors for rows
        successors = []
        for index, row in self.bill_of_materials.iterrows():
            successor = [self.bill_of_materials.columns.get_loc(col) - 1
                         for col in self.bill_of_materials.columns if row[col] == 1]
            successors.append(successor)

        # Vector of vectors for columns
        predecessors = []
        for col in self.bill_of_materials.columns:
            # do not count f
            if col == 'Unnamed: 0':
                continue
            predecessor = [index for index, row in self.bill_of_materials.iterrows() if row[col] == 1]
            predecessors.append(predecessor)

        return successors, predecessors

    def get_capacity_groups(self, df_capacities_groups, component_names):
        capacity_groups = [[] for _ in range(len(df_capacities_groups))]
        for i in range(len(df_capacities_groups)):
            for j in range(len(df_capacities_groups.iloc[i])):
                if pd.isna(df_capacities_groups.iloc[i, j]):
                    continue
                else:
                    capacity_groups[i].append(component_names.index(df_capacities_groups.iloc[i, j]))

        return capacity_groups

    def find_capacity_groups(self):
        find_capacity_group = []
        for i in range(self.num_components):
            find_capacity_group.append([j for j in range(self.num_nodes) if i in self.capacity_groups_indices[j]][0])

        return find_capacity_group

    def get_avg_demand(self):
        avg_demand_per_product = self.product_demand_mean

        avg_demand_per_component = [0 for _ in range(self.num_sub_components)]
        for i in range(self.num_sub_components - 1, -1, -1):
            for suc in self.successors[i]:
                if suc > self.num_sub_components - 1:
                    avg_demand_per_component[i] += avg_demand_per_product[suc - self.num_sub_components]
                else:
                    avg_demand_per_component[i] += avg_demand_per_component[suc]

        avg_demand = avg_demand_per_component + avg_demand_per_product

        return avg_demand
