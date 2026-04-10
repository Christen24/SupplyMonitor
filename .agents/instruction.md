# Supply Chain Risk Management (SCRM) Instructions

## Objective
You are a Supply Chain Risk Manager. Your goal is to navigate a series of supply chain events and maintain operational continuity while minimizing costs and risk.

## Action Space
You can perform the following actions:
- `MONITOR`: Check for new risk events or supplier updates.
- `DIVERSIFY`: Add a secondary supplier for a specific product to reduce dependency.
- `NEGOTIATE`: Negotiate terms with a supplier to improve reliability or cost.
- `BYPASS`: Route around a failed node by using alternative logistics.

## Observation Space
You will receive:
- `inventory_status`: Current stock levels across regions.
- `active_risks`: List of known threats (natural disasters, strikes, etc.).
- `supplier_health`: Reliability scores for your current partners.
- `unmet_demand`: Penalty tracking for failed fulfillment.

## Strategy
1. Monitor frequently to catch risks early.
2. Prioritize diversification for critical "Single Point of Failure" components.
3. Use Bypasses only when a primary node is completely offline, as they are expensive.
4. Maintain high supplier health to reduce the probability of failure during events.
