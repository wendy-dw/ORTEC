class State:
    def __init__(self):
        self.inventory_per_node = []
        self.inventory_in_pipeline_per_node = []

        self.history_demand_per_product = []
        self.aggregate_demand_model_history = []

        # Here you can add additional attributes that you need for your state
