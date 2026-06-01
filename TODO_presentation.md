# TODO — Presentation

## Context

This presentation demonstrates the BCS (Budish, Cramton & Shim 2015) adverse selection
problem in continuous limit order books and evaluates a universal order submission delay
as a structural remedy. The simulation is a discrete-event model of a single-asset
market with three agent types: a market maker, snipers, and investors.

The core argument: continuous trading creates a time window between when a price jump
occurs and when the market maker can react. Any participant who observes the jump faster
can exploit the stale quote. This is not an aberration — it is structural to the design.

---

## Model Components

### 1. Price Process (True Value Y)

Y follows a **jump-diffusion** process over a standard trading day:

- **Jump component**: Poisson arrivals at high frequency throughout the day. Each jump
  is a small discrete number of ticks, with direction (up/down) equally likely and
  larger jumps less probable than smaller ones.
- **Brownian motion component**: A small continuous diffusion term is added at each
  jump interval to give the process realistic long-horizon drift, so full-day charts
  resemble actual price series rather than pure noise.
- After each step, Y is snapped to the nearest tick.

The inter-arrival times between jumps are exponentially distributed (memoryless), so
the jump process is stationary. This is broadly consistent with BCS and the market
microstructure literature on intraday price processes.

---

### 2. Investor Model

Investors represent uninformed, liquidity-motivated order flow.

- Arrivals follow a **Poisson process** at a fixed rate.
- Each investor submits a **market order of unit size** in a randomly chosen direction
  (buy or sell with equal probability).
- Investors have no view on Y and do not optimise — they trade for exogenous reasons.
- They are subject to the same `order_submission_delay` as other participants when
  that parameter is non-zero.

This is the standard noise-trader assumption. Investor flow provides the MM with
spread income that must offset adverse selection losses from snipers.

---

### 3. Sniper Model

Snipers represent fast, informed participants who exploit stale MM quotes.

- Each sniper observes Y with a short lag after each jump — faster than the MM.
- A sniper acts if and only if the best available quote is stale by at least a minimum
  edge threshold relative to the new Y.
- If the condition is met, the sniper submits a **market order** to lift the stale quote.
- When multiple snipers observe the same jump simultaneously, the order in which they
  act is randomised — neither has a systematic advantage over the other.
- **PnL** is computed as mark-to-market at the time of fill: the sniper values the
  position immediately at current Y, so profit equals the gap between the fill price
  and the true value at the moment of execution.
- When `order_submission_delay > 0`, the market order is **deferred**: it arrives at
  the exchange only after the delay elapses. If by that time the MM has repriced, the
  sniper fills at the updated (fair) price and earns no edge.

---

### 4. Market Maker Model

The MM provides continuous two-sided liquidity and updates quotes when it observes Y.

- The MM observes Y with a longer lag than snipers — this lag is the source of the
  sniping vulnerability.
- It posts a **bid and ask symmetrically around a skewed mid**:
  - A fixed base spread is set as the starting point for quote placement.
  - **Inventory skew**: the mid is shifted against the direction of accumulated
    inventory to encourage mean-reversion and limit position risk.
- **Divergence response**: if the current book midpoint has drifted too far from Y,
  the MM cancels both sides and reposts at a wider, defensive spread.
- **Refill on fill**: when a quote is taken, the MM immediately reposts at the same
  price. This keeps the book two-sided but leaves a potentially stale quote in place
  until the next observation event.
- **Cancellations are instantaneous** — this reflects the real market convention
  where exchanges permit immediate cancels to prevent locked books, regardless of any
  submission delay policy.
- When `order_submission_delay > 0`, new limit order submissions and refills are also
  deferred by the same delay. Cancels remain instant.
- **Dynamic spread adjustment (Phase 3)**: the MM tracks a rolling window of fills
  and classifies each as informed or uninformed based on subsequent Y movement.
  It estimates α (fraction of informed flow) and adjusts its spread proportionally,
  inspired by the Glosten-Milgrom framework.

---

## Event Queue and Timing

The simulation is a **discrete-event simulation** driven by a single priority queue.
For each Y jump at time t, three event types are scheduled:

```
t + 0          Y_JUMP          — Y updates; mark-to-market; spread sampled
t + sniper_lag SNIPER_OBSERVE  — each sniper checks for a sniping opportunity
t + mm_lag     MM_OBSERVE      — MM cancels stale quotes and reposts
```

Investor arrivals are interleaved from an independent Poisson stream.

Events with identical timestamps are broken by a random tie-break, so no agent has a
deterministic queue-position advantage over another at the same latency tier.

The **critical window** is `[t + sniper_lag, t + mm_lag]`: snipers can act on the
stale quote before the MM has any chance to update. This window is the mechanism the
BCS paper focuses on, and it is what the order submission delay scenarios attempt to
close.

---

## Presentation Outline

1. **Motivation** — the BCS arms race argument; why continuous LOBs create a structural
   sniping problem independent of regulation or intent.
2. **Model walkthrough** — the four components above, with diagrams of the event timeline.
3. **Scenario 1** — baseline CDA with speed advantage; snipers observe Y faster than MM.
4. **Scenario 2** — equal-speed CDA with many snipers; adverse selection persists without
   any latency advantage, driven purely by competition.
5. **Scenario 3** — small universal submission delay; window not closed, MM still exploited.
6. **Scenario 4** — sufficient universal submission delay; snipers still trade but fill
   at the updated price; edge eliminated and MM PnL recovers.
7. **Conclusion** — quantify the protection the delay provides and its cost to investors
   (spread impact); position relative to BCS batch auction proposal.

See `TODO_sensitivity_analysis.md` for detailed parameter choices per scenario.
