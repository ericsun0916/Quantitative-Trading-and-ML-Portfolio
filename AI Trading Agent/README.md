# 🤖 DMC AI Trading Agent  
### High-Fidelity Market Simulation & Reinforcement Learning Framework

**Project Status**

| Component | Progress |
|---|---|
| Backend Simulation Engine | 🟢 100% |
| Reinforcement Learning Environment | 🟢 100% |
| Frontend Visualization Dashboard | 🟡 90% |
| AI Agent Trainning |  3% |

👉 **[Read the Technical Whitepaper](./agent.pdf)**

---

## 🎥 System Demo

*(Click the image below to watch the frontend dashboard in action, showcasing real-time order flow and AI decision visualization)*

[![AI Agent Frontend Video]](https://app.screencastify.com/watch/qVlATI2ZUjc2Z425FXDL?checkOrg=b36ed518-5720-4ec1-9089-0d17ca87725a)


---
# 📌 Project Overview

This project presents a **high-fidelity market simulation framework** designed to train reinforcement learning agents under realistic trading constraints.

While many quantitative trading models demonstrate impressive backtest performance, they often rely on **simplified market assumptions**, including:

- Infinite liquidity  
- Zero slippage  
- No market impact  
- Implicit look-ahead bias  

Such assumptions significantly distort real-world trading dynamics, frequently producing strategies that fail in live markets.

To address this structural limitation, this project introduces a **fully simulated execution environment** that replicates realistic market conditions. Instead of directly training a model to predict prices, the system trains an **AI trading agent capable of decision-making under market constraints**, similar to professional discretionary traders.

The agent is designed to:

- Understand **multi-timeframe market structure**
- React to **microstructure signals**
- Respect **liquidity and execution constraints**
- Operate under strict **risk management protocols**

---

# 🧠 Core Research Contributions

## 1. Hierarchical Trading Intelligence  
### Decoupling Market Cognition and Execution

Inspired by cognitive dual-process theory (System 1 / System 2), the AI architecture separates **strategic reasoning** from **trade execution**.

### Manager (Strategic Layer — System 2)

Responsible for macro-level decision making:

- Interprets **4H / Daily market structure**
- Maintains an external dynamic memory system (**The Notebook**)
- Classifies market regimes:
  - Trending
  - Ranging
  - Breakout

This layer establishes the **directional bias and trading context**.

---

### Worker (Execution Layer — System 1)

Handles microstructure-level trade execution:

- Observes **Order Flow Imbalance (OFI)**
- Detects **price-volume divergences**
- Executes entries and exits at **Tick / 1-minute resolution**

The Worker operates strictly within the constraints defined by the Manager.

---

### IQN Risk Gatekeeper

An **Implicit Quantile Network (IQN)** module evaluates extreme tail risk:

- Estimates **left-tail return distributions**
- Detects abnormal volatility regimes
- **Vetoes high-risk actions** during potential Black Swan scenarios

This component acts as a **risk firewall for the trading agent**.

---

# ⚙️ High-Fidelity Execution Simulator

A custom execution engine was built to replicate **realistic derivatives trading environments**, eliminating common simplifications used in traditional backtesting frameworks.

### Square-Root Slippage Model

The simulator incorporates a **non-linear market impact function**:

- Slippage scales with **order size**
- Adjusted by **real-time volatility**
- Adjusted by **daily traded volume**

This follows empirical findings observed in institutional execution research.

---

### Dynamic Margin & Liquidation Engine

The simulator faithfully replicates **Binance Futures tiered margin mechanics**, including:

- Tiered leverage limits
- Dynamic maintenance margin
- Forced liquidation triggers

This ensures the RL agent learns to operate under **real leverage constraints**, preventing unrealistic capital usage.

---

# 🎓 Curriculum Learning Framework

To stabilize reinforcement learning training and emulate human learning progression, the agent is trained through **four structured phases**.

## Phase 0 — Observer

The agent does not trade.

Focus:

- Learn **market structure**
- Identify **support and resistance zones**

---

## Phase 1 — Cruise

The agent learns **discipline and patience**.

Objective:

- Avoid low-liquidity periods
- Recognize **high-quality trading windows**

---

## Phase 2 — Soldier

Strict **trend-following regime**.

Rules:

- Only trade in trend direction
- Counter-trend actions receive heavy penalties

---

## Phase 3 — Master

Full trading capability unlocked.

Agent can:

- Execute **counter-trend trades**
- Hedge positions
- Trade **mean-reversion at key liquidity zones**

---

# 📊 Frontend Visualization Dashboard

**Current Progress: ~90%**

Relevant branches:

```
frontend_replay
frontend_frvp
frontend_delta_footprint
frontend_basic_vue
```

Early development attempts used **TSX-based chart rendering**, but the approach struggled with:

- high-frequency tick data
- complex AI decision visualization
- extensibility limitations

The frontend architecture was therefore redesigned to support **professional-grade financial visualization**.

### Current Capabilities

- Real-time **K-line chart rendering**
- Visualization of **AI memory zones ("The Notebook")**
- **Order flow heatmaps**
- Interactive **AI attention weight visualization**

The ability to inspect **model attention weights** is particularly important for:

- RL training diagnostics
- strategy validation
- post-training interpretability

---

# 🧩 Technology Stack

## Core Engine

- **Rust** — high-performance simulation backend  
- **Python** — RL training pipeline  
- **Polars** — lazy evaluation data pipeline  

---

## Reinforcement Learning

- **PPO (Proximal Policy Optimization)**
- **LSTM**
- **IQN (Implicit Quantile Networks)**
- **PyTorch**

---

## Frontend

- **Vue.js**
- Advanced financial charting libraries

---

## Data Sources

Historical data sourced from **Binance Futures**:

- Tick-level **aggTrades**
- OHLCV market data

Dataset period:

```
2020 — 2025
```

---

# 📄 Additional Documentation

For detailed technical explanations, please refer to the full whitepaper:

👉 **DMC_III_final.pdf**

The document includes:

- Slippage model mathematical formulation
- Reward function design
- Time-regime encoding methodology
- RL training architecture
- evaluation results

---
## 💻 Code Highlights

Here is a glimpse into the core engine detailing how the system handles realistic execution and hierarchical RL actions:

### 1. Square-Root Slippage Implementation
Discarding simple fixed-percentage slippage, the engine calculates real-time market impact based on order size and local volatility to prevent the agent from exploiting infinite liquidity.

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

    // [核心機制] 清算瀑布發生時，市場衝擊係數放大 10 倍
    let active_c = if is_cascade {
        self.impact_coefficient * 10.0
    } else {
        self.impact_coefficient
    };

    let effective_volatility = volatility * urgency_multiplier;
    // Square-Root Law of Market Impact
    let impact = active_c * effective_volatility * participation_rate.sqrt();
    
    // 鎖定最高滑點 5% 防止模擬崩潰
    (impact + spread_cost).min(0.05) 
}
```

### 2. Microstructure Feature Extraction
Standard OHLCV bars obscure critical market dynamics. The LiveBarGenerator aggregates high-frequency tick data in real-time, extracting Order Flow Imbalance (OFI) and sub-millisecond time deltas to feed the AI's System 1 reflex layer.

```rust
// src/live_bar.rs
pub struct LiveBarState {
    pub timestamp: i64,      // 毫秒時間戳
    pub close: f64,
    pub volume: f64,
    
    // 微觀結構特徵 (Microstructure Features)
    pub buy_vol: f64,          // 主動買盤量 (Buyer Maker = False)
    pub sell_vol: f64,         // 主動賣盤量 (Buyer Maker = True)
    pub volume_imbalance: f64, // 同一毫秒內的極端累積成交量
    pub time_delta: f64,       // 距離上一筆 Tick 的真實時間差 (ms)
}
```

### 3. Graceful Degradation in Data Pipeline
To robustly handle vast amounts of historical data, the pipeline features a graceful degradation fallback. It reconstructs ultra-precise Footprint charts when tick data is available, and automatically falls back to an OHLCV uniform distribution when it's not.

```rust
// src/data_loader.rs
if t_end > t_start {
    // 🚀 情況 A：具備高擬真 Tick 數據，使用真實 Order Flow 精確還原
    let bar = FootprintBar::from_ticks(
        st_time, k_open[i], k_high[i], k_low[i], k_close[i],
        &t_px[t_start..t_end], &t_qty[t_start..t_end], &t_bm[t_start..t_end],
        dynamic_bin_size
    );
    target_bars.push(bar);
} else {
    // 🛡️ 情況 B：無 Tick 數據 (例如歷史區間)，啟動 OHLCV 均勻分配降級算法
    if k_vol[i] > 0.0 {
        let bar = FootprintBar::from_ohlcv(
            st_time, k_open[i], k_high[i], k_low[i], k_close[i],
            k_vol[i], dynamic_bin_size
        );
        target_bars.push(bar);
    }
}
```

### 4. Time-Scaled WebSocket Engine
For the frontend to accurately visualize AI training and historical replays, the WebSocket server implements a time-throttle engine. It dynamically scales the asynchronous sleep duration based on true historical tick deltas and user-defined playback speeds.

```rust
// src/websocket.rs
// 【時間節流閥核心】動態縮放回放時間
if has_next {
    // real_time_delta_ms 為歷史 Tick 之間的真實毫秒差
    let scaled_wait_ms = (real_time_delta_ms / playback_speed).max(1.0);
    sleep_duration = Duration::from_millis(scaled_wait_ms as u64);
    
    // 透過非同步 sleep 控制推播節奏，實現真實感 Replay
    sleep(sleep_duration).await;
} else {
    is_playing = false;
}
```

### 5. Perpetual Futures Mechanics & Funding Fees
The simulator strictly adheres to Binance Futures mechanics, including funding rate settlements. This forces the RL agent to incorporate holding costs and premium index arbitrage into its long-term strategic decision-making.

```rust
// src/account.rs
pub fn apply_funding_fee(&mut self, premium_index: f64, mark_price: f64) -> f64 {
    let mut total_fee = 0.0;
    let base_interest = 0.0001; 
    let funding_rate = premium_index + base_interest; 

    if let Some(pos) = &self.long_position {
        let notional = pos.size * mark_price;
        // 多頭在資金費率為正時支付費用
        let fee = notional * funding_rate; 
        self.balance -= fee;
        total_fee -= fee;
    }
    // ... Short position logic ...
    
    self.update_pnl(mark_price, mark_price);
    total_fee
}
```

### 6. High-Performance Async API (Polars)
The backend utilizes Rust's axum framework for zero-cost async routing. Memory-safe shared states and Polars DataFrames are used to calculate complex Fixed Range Volume Profiles (FRVP) on the fly with sub-millisecond latency.

```rust
// src/handlers.rs
pub async fn get_frvp(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<FrvpRequest>,
) -> Json<Value> {
    let session = state.session.lock().await;

    // 直接在記憶體中利用 Polars (Dataframe) 的底層結構進行極速 FRVP 聚合
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

💡 **Note:**  
This repository focuses on the **engineering implementation and framework design**.  
For the complete research discussion and mathematical derivations, please consult the whitepaper.