# DMC AI Trading Agent
### Execution-Aware Market Simulation & Reinforcement Learning Research Framework

## Project Status

| Component | Status |
|---|---|
| Market Simulation Engine | Completed |
| Reinforcement Learning Environment | Completed |
| Frontend Visualization Dashboard | Functional / Iterating |
| Agent Training & Evaluation | Ongoing |

👉 **[Read the Technical Whitepaper](./agent.pdf)**

---

## System Demo

*(Click the image below to watch the frontend dashboard in action, including historical replay, order-flow visualization, and model diagnostics.)*

[![AI Agent Frontend Video]](https://app.screencastify.com/watch/qVlATI2ZUjc2Z425FXDL?checkOrg=b36ed518-5720-4ec1-9089-0d17ca87725a)

---

# Research Motivation

This project began from a practical trading question:

> **Can discretionary trading techniques be translated into explicit rules, quantitative signals, and programmable decision processes?**

The project was motivated by trading methods I had learned through market study, trader-led review sessions, and practical trading experience. These included **multi-timeframe market structure, support and resistance, VWAP, order-flow interpretation, trade planning, position management, and risk control**.

Rather than using these ideas only as discretionary chart-reading tools, I wanted to formalize them into a system that could be:

- expressed as measurable features and decision rules,
- tested repeatedly across historical market conditions,
- evaluated under explicit risk constraints,
- and eventually incorporated into an automated decision framework.

This led to a second research problem: **the trading environment itself must be realistic enough for model evaluation to be meaningful**.

A strategy trained under assumptions such as infinite liquidity, zero slippage, no transaction costs, or unrealistic margin mechanics can learn behavior that would not survive practical execution. The project therefore evolved from a trading-agent idea into a broader **execution-aware market simulation and reinforcement-learning framework**.

The objective is not to apply AI to trading for its own sake. The objective is to study whether trading logic learned from discretionary practice can be represented consistently in a quantitative system and evaluated under realistic market constraints.

---

# Project Overview

The framework combines three research components:

1. **Trading logic formalization**  
   Translating discretionary trading concepts into measurable market features and explicit decision states.

2. **Execution-aware market simulation**  
   Modeling slippage, transaction costs, margin requirements, liquidation mechanics, and liquidity constraints.

3. **Reinforcement-learning experimentation**  
   Testing whether an agent can learn decision policies within these constraints rather than simply predicting short-term prices.

The system is designed to work with:

- **multi-timeframe market structure**
- **order-flow and microstructure signals**
- **liquidity and execution constraints**
- **position and risk management rules**

---

# System Architecture

## 1. Hierarchical Decision Framework
### Separating Market Context from Execution

The architecture separates higher-level market interpretation from lower-level execution decisions.

### Manager — Strategic Layer

The strategic layer interprets broader market context:

- **4H / Daily market structure**
- support / resistance zones
- market regime classification
- directional context
- external state memory through **The Notebook**

The current regime framework includes:

- Trending
- Ranging
- Breakout

The purpose of this layer is to define the broader decision context rather than trigger individual orders directly.

---

### Worker — Execution Layer

The execution layer handles lower-timeframe trade decisions using information such as:

- **Order Flow Imbalance (OFI)**
- price-volume relationships
- short-horizon market activity
- position state
- execution constraints

It operates at **tick / 1-minute resolution** and remains constrained by the strategic context defined by the higher-level layer.

This separation is intended to reduce the tendency of a single model to mix slow market-context decisions with high-frequency execution behavior.

---

### Experimental Tail-Risk Module

An **Implicit Quantile Network (IQN)** component is included as an experimental risk layer.

Its intended role is to:

- estimate the lower tail of the return distribution,
- identify abnormal volatility states,
- and restrict selected actions when modeled downside risk becomes unusually high.

This module remains part of the ongoing training and evaluation work and should be viewed as an experimental risk-control component rather than a fully validated production risk model.

---

# Execution-Aware Market Simulator

The simulation engine was developed to avoid several simplifying assumptions common in basic backtests.

The current framework includes:

- transaction costs,
- slippage,
- order-size effects,
- volatility-sensitive market impact,
- leverage constraints,
- maintenance margin,
- liquidation logic,
- and funding fees.

The purpose is not to reproduce every detail of a live exchange, but to create a research environment in which strategy behavior is evaluated under more realistic execution and risk constraints.

---

## Square-Root Slippage Model

The simulator uses a non-linear slippage model in which estimated market impact depends on:

- order size,
- traded volume,
- volatility,
- and execution urgency.

The model is inspired by the empirical square-root relationship commonly used in market-impact research.

This is intended to prevent the agent from exploiting the unrealistic assumption of unlimited liquidity.

---

## Margin & Liquidation Engine

The simulator models **Binance-style perpetual-futures mechanics**, including:

- tiered leverage limits,
- maintenance margin,
- forced liquidation triggers,
- and funding settlements.

These constraints force the agent to account for leverage and holding costs when evaluating trading actions.

The implementation is a research approximation of exchange mechanics rather than a claim of exact exchange-level replication.

---

# Curriculum Learning Framework

The training design uses four stages to gradually increase decision complexity.

## Phase 0 — Observer

The agent does not trade.

Focus:

- identifying market structure,
- learning support / resistance zones,
- observing market-state transitions.

---

## Phase 1 — Cruise

The agent begins learning selective participation.

Focus:

- avoiding low-quality market conditions,
- recognizing higher-quality trading windows,
- maintaining position discipline.

---

## Phase 2 — Soldier

The agent operates under a trend-following constraint.

Rules:

- trade only in the dominant trend direction,
- penalize counter-trend actions,
- reinforce consistent risk behavior.

---

## Phase 3 — Master

The action space becomes broader.

The agent may:

- take counter-trend positions,
- hedge exposure,
- trade mean-reversion setups near defined liquidity zones.

This stage is intended for later experimentation after the simpler policy stages are stable.

---

# Frontend Visualization Dashboard

The visualization layer supports historical replay, strategy inspection, and model diagnostics.

Relevant branches:

```text
frontend_replay
frontend_frvp
frontend_delta_footprint
frontend_basic_vue
```

### Current Capabilities

- K-line chart rendering
- historical market replay
- visualization of **AI memory zones ("The Notebook")**
- order-flow heatmaps
- strategy-state visualization
- model-diagnostic views

The frontend is designed primarily as a research and debugging interface rather than as a production trading terminal.

---

# Technology Stack

## Core Engine

- **Rust** — simulation backend
- **Python** — reinforcement-learning and research pipeline
- **Polars** — market-data processing
- **Parquet** — historical data storage

## Reinforcement Learning

- **PPO (Proximal Policy Optimization)**
- **LSTM**
- **IQN (Implicit Quantile Networks)**
- **PyTorch**

## Frontend

- **Vue.js**
- financial charting libraries

## Data Sources

Historical data sourced from **Binance** includes:

- tick-level **aggTrades**
- OHLCV market data
- perpetual-futures related inputs used by the simulator

Dataset period:

```text
2020 — 2025
```

---

# Additional Documentation

For detailed technical explanations, please refer to the full whitepaper:

👉 **[Technical Whitepaper](./agent.pdf)**

The whitepaper covers topics including:

- slippage-model formulation,
- reward-function design,
- time-regime encoding,
- reinforcement-learning architecture,
- and evaluation methodology.

---

# Code Highlights

Below are selected implementation examples from the simulation framework.

## 1. Square-Root Slippage Implementation

The engine estimates market impact as a function of order participation and local volatility rather than using a fixed-percentage slippage assumption.

```rust
// src/slippage.rs
pub fn calculate_slippage(
    &self,
    order_size_notional: f64,
    daily_volume: f64,
    volatility: f64,
    urgency_multiplier: f64,
    is_cascade: bool,
) -> f64 {
    let spread_cost = self.base_spread / 2.0;
    let participation_rate = order_size_notional / daily_volume;

    let active_c = if is_cascade {
        self.impact_coefficient * 10.0
    } else {
        self.impact_coefficient
    };

    let effective_volatility = volatility * urgency_multiplier;
    let impact = active_c * effective_volatility * participation_rate.sqrt();

    (impact + spread_cost).min(0.05)
}
```

---

## 2. Microstructure Feature Extraction

The `LiveBarGenerator` aggregates tick data into lower-frequency states while preserving selected order-flow information.

```rust
// src/live_bar.rs
pub struct LiveBarState {
    pub timestamp: i64,
    pub close: f64,
    pub volume: f64,

    pub buy_vol: f64,
    pub sell_vol: f64,
    pub volume_imbalance: f64,
    pub time_delta: f64,
}
```

These features are used to expose lower-timeframe market activity to the execution layer.

---

## 3. Graceful Degradation in the Data Pipeline

When tick data are available, the system reconstructs footprint-style information directly from trades. When only OHLCV data are available, it falls back to an approximate bar-level representation.

```rust
// src/data_loader.rs
if t_end > t_start {
    let bar = FootprintBar::from_ticks(
        st_time, k_open[i], k_high[i], k_low[i], k_close[i],
        &t_px[t_start..t_end], &t_qty[t_start..t_end], &t_bm[t_start..t_end],
        dynamic_bin_size
    );
    target_bars.push(bar);
} else {
    if k_vol[i] > 0.0 {
        let bar = FootprintBar::from_ohlcv(
            st_time, k_open[i], k_high[i], k_low[i], k_close[i],
            k_vol[i], dynamic_bin_size
        );
        target_bars.push(bar);
    }
}
```

The fallback is intentionally approximate and should not be interpreted as a substitute for true tick-level order-flow data.

---

## 4. Time-Scaled WebSocket Replay

The WebSocket layer preserves historical time spacing while allowing accelerated replay for research and debugging.

```rust
// src/websocket.rs
if has_next {
    let scaled_wait_ms = (real_time_delta_ms / playback_speed).max(1.0);
    sleep_duration = Duration::from_millis(scaled_wait_ms as u64);
    sleep(sleep_duration).await;
} else {
    is_playing = false;
}
```

---

## 5. Perpetual-Futures Funding Logic

The simulator incorporates funding settlements so that longer holding periods are affected by ongoing position costs.

```rust
// src/account.rs
pub fn apply_funding_fee(&mut self, premium_index: f64, mark_price: f64) -> f64 {
    let mut total_fee = 0.0;
    let base_interest = 0.0001;
    let funding_rate = premium_index + base_interest;

    if let Some(pos) = &self.long_position {
        let notional = pos.size * mark_price;
        let fee = notional * funding_rate;
        self.balance -= fee;
        total_fee -= fee;
    }

    // ... Short position logic ...

    self.update_pnl(mark_price, mark_price);
    total_fee
}
```

This implementation is part of the research simulator and is intended to capture the economic effect of funding rather than provide an exact exchange-matching engine.

---

## 6. Async Research API

The backend uses Rust's `axum` framework and Polars DataFrames to support interactive research calculations such as Fixed Range Volume Profile (FRVP).

```rust
// src/handlers.rs
pub async fn get_frvp(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<FrvpRequest>,
) -> Json<Value> {
    let session = state.session.lock().await;

    match DataLoader::calculate_frvp_in_memory(
        payload.start_ts, payload.end_ts, payload.va_ratio,
        &session.timestamps, &session.closes, &session.volumes,
        session.current_ticks.as_ref()
    ) {
        Ok(profile) => Json(json!({ "status": "success", "data": profile })),
        Err(e) => Json(json!({ "status": "error", "message": e.to_string() }))
    }
}
```

---

# Research Limitations

This project is still under active development.

Important limitations include:

- agent training and out-of-sample policy evaluation are still ongoing,
- simulated execution cannot fully reproduce live exchange behavior,
- slippage and impact functions depend on modeling assumptions,
- Binance historical trade data do not reveal the complete hidden-liquidity state of the market,
- some fallback order-flow representations are approximations,
- reinforcement-learning performance must be evaluated carefully for overfitting and regime dependence.

These limitations are part of the research question rather than something the framework attempts to hide.

---

# Current Research Direction

The next stage of the project focuses on:

- completing agent training,
- comparing learned policies with simpler rule-based baselines,
- testing stability across different market regimes,
- evaluating turnover and execution costs,
- analyzing whether the agent is learning economically interpretable behavior,
- and distinguishing genuine policy improvement from simulator-specific overfitting.

The central question remains:

> **Can discretionary trading logic be formalized into a quantitative system without losing the market context and risk discipline that make the original trading framework useful?**

---

## Repository Note

This repository focuses on the **engineering implementation, market-simulation framework, and research design**.

The project should be viewed as an ongoing quantitative trading experiment rather than a production-ready automated trading system.
