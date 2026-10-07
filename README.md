# Trading & Quantitative Research Portfolio

I focus on **trading, quantitative research, and market modeling**, with particular interests in **derivatives, volatility, market structure, systematic strategies, and execution**.

My work combines financial modeling, empirical research, and programming to study how markets are priced, how trading signals behave, and how strategies perform under realistic risk and execution constraints.

## Research Interests

- Derivatives & Volatility
- Market Structure & Execution
- Systematic Trading
- Quantitative Modeling
- Risk & Portfolio Behavior
- Financial Machine Learning

---

# Selected Research

## 1. USD/JPY Gamma Scalping & Volatility Research

**Domain**  
Options · Volatility · Dynamic Hedging · Risk

**Overview**

A quantitative study of a **USD/JPY long-volatility strategy** built around gamma scalping, implied-volatility dislocations, and dynamic delta hedging.

The research treats the strategy as a volatility-driven relative-value problem rather than a directional FX trade. It examines whether option convexity and repeated spot rebalancing can generate sufficient realized gains to offset theta decay and implementation frictions.

**Key Focus**

- Long straddle / long-volatility structure
- Gamma, theta, and vega interpretation
- Dynamic delta hedging
- Volatility dislocation framework
- Bid-ask spread, slippage, and transaction costs
- Risk-adjusted performance and drawdown analysis

👉 [View Research](./usdjpy/README.md)

---

## 2. AI Trading Agent & Execution-Aware Market Simulation

**Domain**  
Quantitative Trading · Market Simulation · Reinforcement Learning · Execution

**Research Motivation**

This project began from a practical question:

> **Can the discretionary trading techniques I had learned be translated into explicit rules, quantitative signals, and programmable decision processes?**

The goal was not to apply AI to trading for its own sake. I wanted to take trading ideas I had learned and used in practice—such as **multi-timeframe market structure, support/resistance, VWAP, order-flow interpretation, position management, and risk control**—and formalize them into a system that could be tested, simulated, and eventually automated.

That led to a second problem: before training an agent, the trading environment itself had to be realistic enough to make the results meaningful. A model trained with infinite liquidity, zero slippage, or unrealistic margin assumptions can learn behavior that would not survive in real markets.

The project therefore evolved into an **execution-aware market simulation and reinforcement-learning research framework**.

**Framework**

The system separates two levels of decision-making:

- **Strategic layer:** interprets higher-timeframe market structure and defines directional context
- **Execution layer:** reacts to lower-timeframe market information and manages entries, exits, and position adjustments

The simulator explicitly models practical trading constraints including:

- Slippage and liquidity effects
- Transaction costs
- Margin requirements
- Liquidation mechanics
- Multi-timeframe information
- Order-flow and market-structure signals

**Technical Implementation**

- **Rust** for the simulation backend
- **Python** for the reinforcement-learning pipeline
- **Polars / Parquet** for market-data processing
- Binance market data for execution and microstructure research
- Visualization tools for strategy review and model diagnostics

The current focus is on improving the simulation environment and evaluating whether trading logic learned from discretionary practice can be represented consistently in a quantitative framework.

👉 [View Project](./AI%20Trading%20Agent/README.md)

---

## 3. Multi-Market Factor Research

**Domain**  
Systematic Trading · Factor Research · Financial Machine Learning

**Overview**

A cross-asset research project covering **30+ instruments** across US equities, ETFs, and cryptocurrencies.

The objective is to study whether structurally motivated market features can provide useful predictive information while maintaining better downside characteristics than a purely return-maximizing approach.

**Key Focus**

- Custom factor design
- Volatility-adjusted momentum
- Price-volume relationships
- Cross-asset feature analysis
- Random Forest / XGBoost comparison
- SHAP-based model interpretation
- Backtesting and drawdown analysis

Two custom factors were developed:

- **VARM — Volatility-Adjusted Relative Momentum**
- **PVD — Price-Volume Divergence**

The project emphasizes understanding **why** a model behaves the way it does, rather than treating prediction accuracy alone as evidence of a robust trading edge.

👉 [View Research](./Machine%20Learning%20Multi-Asset%20Factor%20Research/README.md)

---

# Other Quantitative Work

## Credit Risk Prediction & Cost-Sensitive Learning

**Domain**  
Risk Modeling · Classification

A machine-learning project using a dataset of **30,000 credit-card clients** to study default prediction under severe class imbalance.

The project focuses on interpretable financial stress features, cost-sensitive learning, and improving recall for high-risk observations.

👉 [View Project](./Credit%20Risk%20Prediction%20%26%20Cost-Sensitive%20Learning/README.md)

---

# Tools & Methods

**Programming**  
Python · Rust · SQL · JavaScript

**Quantitative Research**  
Time-Series Analysis · Factor Research · Backtesting · Reinforcement Learning · Model Interpretation

**Market & Execution Research**  
Derivatives · Volatility · Market Structure · Liquidity · Slippage · Risk Management

**Data & Systems**  
Polars · Pandas · Parquet · PostgreSQL · REST APIs · Event-Driven Architecture

---

# Research Principles

Across these projects, I focus on:

- **Economic intuition before model complexity**
- **Research reproducibility**
- **Realistic execution assumptions**
- **Risk-aware evaluation**
- **Model interpretability**
- **Clear separation between backtest results and live-trading evidence**

The goal is not simply to build models that look predictive in-sample, but to understand **why a strategy may work, where it can fail, and whether its assumptions remain defensible under real trading constraints**.

---

# Repository Scope

This repository contains selected **independent and publicly shareable research projects**.

Ongoing academic research, unpublished work, and employer-related research are intentionally excluded where appropriate.

---

# Contact

If you're interested in **trading, quantitative research, derivatives, or systematic market analysis**, feel free to reach out.
