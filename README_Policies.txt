This readme file explains what each policy does.

Authors: Tim Schiettecatte, Kian Pessendorffer, Wendy Dewit, Lieve Dewallef

May 2025

----------------------------------------------------------------------------------------------------------------------------------

Already provided policies (3): MeanDemandPolicy, StaticBaseStockPolicyShortfall, and StaticBaseStockPolicyRandom. 
Created policies (6): EchelonBaseStockPolicy, NonstationaryBaseStockPolicy, sSPolicy, DualIndexPolicy, DynamicPolicy, and StaticPolicy 

All policies take production capacity and availability of needed materials into account. Furthermore, all policies use a hedging parameter as a safety net against shortage, which can be adapted.

----------------------------------------------------------------------------------------------------------------------------------

MeanDemandPolicy simply uses the mean demand for production. Decisions on which parts to make are generated in a random order. 
StaticBaseStockPolicyShortfall defines base-stock levels for each component based on how long it takes to make that part (i.e. lead times) and the average demand. These base-stock levels are the target inventory values to minimise shortfalls. For production, the current inventory levels are taken into account, as well as the parts that use other parts for production. It furthermore prioritises bottlenecks in terms of parts that can cause shortages further in the supply chain when parts cannot be produced simultaneously. Other nodes are handled randomly.

StaticBaseStockPolicyRandom is similar to StaticBaseStockPolicyShortfall in how it defines base-stock levels, with the addition that deviations can be provided during initialisation. This way, for example, promotions can be prepared for be defining that more of certain nodes should be produced. Furthermore, this policy does not prioritise bottleneck nodes like StaticBaseStockPolicyShortfall does, but it handles each node equally in random order.

EchelonBaseStockPolicy is similar to both StaticBaseStockPolicies. It calculates the target inventory levels based on lead times and average demands. These target inventory levels are held constant over time. However, components are always processed in the same order, not defined by priority reasons.

NonstationaryBaseStockPolicy also calculates the target inventory levels based on lead times and average demands. These levels change over time using forecasting of future demands based on historical demand data. A safety stock component is included based on the desired frequency of meeting demand (service level) and expected variability in the demand forecast over the lead time.
sSPolicy defines two inventory levels. Small s is the lower threshold or the reorder point. Suppose the inventory goes to or below s. In that case, the policy aims to replenish the inventory up to big S. Both when checking the inventory and when calculating how much to order, the current inventory level and what has already been ordered or is still in transit are taken into account. Both s and S have to be defined for this policy to work.

DualIndexPolicy also uses small s (lower level) and big S (upper level), which have to be defined. The lower level is based on the on-hand inventory, and the upper level is the sum of the on-hand inventory and what has been ordered or is in transit. When a reorder is triggered by the on-hand inventory going below s, a decision to order is made, taking into account the on-hand inventory and what has been ordered or is in transit.

DynamicPolicy uses the average demand adapted by the hedging parameter as a starting point, which gets adjusted based on several factors. It takes lead times into account. The hedging parameter gets adjusted dynamically (i) globally based on average demand variability and (ii) for each component, based on the frequency of past shortages for. When there is a shortage in the same supply for several nodes, nodes with low stock are given priority. Then, components required for end products with the largest deviations from their target levels receive higher priority.

StaticPolicy is highly similar to DynamicPolicy, but the hedging parameter is static.

----------------------------------------------------------------------------------------------------------------------------------