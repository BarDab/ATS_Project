# TODO

## Configurable order delay
Add configurable latency for all order types (limit submissions, market orders, refills)
besides cancellations. Cancels should remain instantaneous — this reflects the real
market structure where exchanges allow fast cancels to prevent locked books.

## Sensitivity analysis — spread vs delayed orders
How much does the MM need to adjust its spread to achieve the same profits as in the
scenario where snipers face delayed orders? Quantify the trade-off between structural
protection (order delay) and price compensation (wider spread).

## Monte Carlo
Run the simulation across many seeds to get distribution of outcomes (MM PnL, alpha,
avg spread, inventory volatility). Current results are single-seed — need to verify
findings are robust and not seed-specific.

## Queue randomization (longer term)
To properly model queue position and speed races we would need:
- At least one more MM in the model (competing for queue priority)
- Order sizes beyond 1 (varied lot sizes)
- Different connection speeds for different MMs

The goal would be to show that if all MMs invest in speed, the arms race equilibrium
forces them to reflect those infrastructure costs in their spreads — a wider spread
with no improvement in liquidity quality. This is hard to model cleanly but would
directly support the BCS argument for batch auctions.
