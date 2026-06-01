# TODO

## Sensitivity Analysis — CDA vs Delayed Order Submission

The goal is to demonstrate that the BCS adverse selection problem exists in two distinct
forms (speed-based and competition-based), and that a universal order submission delay
can eliminate sniper profitability without requiring batch auctions.

All scenarios report: MM realized PnL, average spread, investor fill count, sniper fill
count, sniper PnL, and MM inventory path. Run each over multiple seeds (Monte Carlo)
to confirm findings are not seed-specific.

---

### Scenario 1 — Baseline CDA: Speed Advantage (1 MM, 2 Snipers)

**Params**: `mm_lag=10ms`, `sniper_lag=1ms`, `n_snipers=2`, `order_submission_delay=0`

The canonical BCS setup. Snipers observe Y jumps 9ms before the MM can react and pick
off stale quotes in the [t+1ms, t+10ms] window. This establishes the baseline cost of
adverse selection — how much the MM loses to informed flow vs. the no-sniper benchmark,
and what spread it must charge to survive.

---

### Scenario 2 — Baseline CDA: Competition Effect (1 MM, 10 Snipers, Equal Speed)

**Params**: `mm_lag=1ms`, `sniper_lag=1ms`, `n_snipers=10`, `order_submission_delay=0`

Speed advantage is removed (MM and snipers observe Y simultaneously), but the large
sniper population means that on every jump, many snipers race to fill the same stale
quote. Even with no latency edge, sheer competition ensures stale quotes get picked off
reliably. This isolates the *structural* adverse selection from the speed component:
the problem persists even in a hypothetical world where all participants are equally fast.

---

### Scenario 3 — Delayed Submission: Insufficient Delay

**Params**: `mm_lag=10ms`, `sniper_lag=1ms`, `n_snipers=2`,
`order_submission_delay` set so that `sniper_lag + order_submission_delay < mm_lag`
(e.g. 5ms delay → sniper orders land at t+6ms, still before MM updates at t+10ms)

All participants face a positive order submission delay (limit orders, market orders,
refills), but the delay is too small to close the sniping window. Snipers still fill
at the stale price and extract the same edge. MM is exploited at roughly the same rate.
Establishes that partial delays are ineffective — they impose friction on everyone
without solving the adverse selection problem.

---

### Scenario 4 — Delayed Submission: Sufficient Delay (Edge Eliminated)

**Params**: `mm_lag=10ms`, `sniper_lag=1ms`, `n_snipers=2`,
`order_submission_delay` set so that `sniper_lag + order_submission_delay >= mm_lag`
(e.g. 10ms delay → sniper orders land at t+11ms, after MM has already updated at t+10ms)

The universal submission delay forces sniper market orders to arrive *after* the MM has
cancelled or repriced the stale quote. Snipers still execute — their orders fill against
the MM's updated quotes — but they no longer get the stale price they acted on. The
fill price reflects current Y, so their per-trade edge drops to zero and sniper PnL
collapses. Show that MM PnL and spread converge toward the no-sniper benchmark from
Scenario 1, quantifying the protection the delay provides. The cost side (wider spreads
for investors due to higher friction) should also be measured.

---

## Monte Carlo

Run each scenario across many seeds to get distributions of MM PnL, average spread,
sniper fill count, sniper PnL, and inventory volatility. Current results are single-seed
— need to verify findings are robust.

## Queue Randomization (Longer Term)

To properly model queue position and speed races:
- At least one more MM competing for queue priority
- Varied lot sizes beyond 1
- Different connection speeds per MM

Goal: show that if all MMs invest in speed, the arms race equilibrium forces wider
spreads with no improvement in liquidity quality — directly supporting the BCS argument.
