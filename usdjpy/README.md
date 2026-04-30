# USD/JPY Gamma Scalping Research

## Overview

This project presents a quantitative research study of an institutional-style USD/JPY options strategy built around **gamma scalping**, **volatility mean reversion**, and **dynamic delta hedging**.  
The core idea is to identify periods when implied volatility appears undervalued relative to an internal reference framework, enter a long-volatility position, and monetize subsequent spot movement through systematic rebalancing of delta exposure.

Rather than framing the strategy as a simple directional trade, this research treats it as a **volatility-driven relative value strategy**. The return generation mechanism relies on the interaction between option convexity, spot re-hedging, and changes in implied volatility, while explicitly accounting for implementation frictions such as spread, slippage, and transaction costs.

This repository is a **research summary version** of the project. Certain implementation details, parameter settings, and execution rules are intentionally omitted.

---

## Research Objective

The project was designed to study three main questions:

1. Can volatility dislocations in the USD/JPY options market be systematically identified and converted into tradeable opportunities?
2. Can a dynamic delta-hedging engine extract enough realized movement from the underlying spot market to offset option decay and trading frictions?
3. What kind of return profile, risk structure, and Greeks exposure emerge when the strategy is tested under realistic assumptions?

The broader goal is not only to test whether the strategy makes money historically, but also to understand **why** it performs the way it does, where the return comes from, and what structural risks remain embedded in the portfolio.

---

## Strategy Logic

The strategy is built around a **long straddle + dynamic hedge** framework.

When market-implied volatility appears sufficiently cheap relative to an internal volatility benchmark, the system opens a long-volatility position through an at-the-money straddle structure.  
Once the options position is established, the directional delta exposure is continuously monitored and offset in the USD/JPY spot market through a rule-based hedging engine.

From a portfolio perspective, the strategy is designed to be:

- **Long Gamma**: allowing spot movement to create re-hedging opportunities.
- **Long Vega**: benefiting from favorable implied volatility expansion.
- **Short Theta in carry terms**: requiring realized price movement or volatility repricing to overcome time decay.

This creates a payoff structure where profitability depends not on forecasting directional spot moves, but on whether the market delivers enough movement and/or volatility repricing to compensate for premium decay and execution costs.

---

## Methodology

### 1. Volatility Mispricing Framework

The entry logic is based on the idea that short-term market-implied volatility can temporarily diverge from a smoother internal estimate of “fair” volatility.  
Instead of using a naive historical volatility measure, the model relies on an internal benchmark derived from recent implied volatility behavior, allowing the strategy to focus on relative dislocation rather than absolute volatility level alone.

The strategy does **not** blindly buy volatility whenever IV is low.  
It only activates when the gap between internal reference volatility and market pricing becomes large enough to justify both option premium exposure and the expected cost of hedging.

### 2. Dynamic Delta Hedging

Once a position is open, the core engine dynamically adjusts spot exposure in response to changing portfolio delta.  
This converts the positive gamma profile of the options position into realized trading gains when the spot market oscillates.

A key part of the design is avoiding excessive micro-rebalancing.  
In practice, over-hedging can destroy theoretical edge because every rebalance adds spread, slippage, and execution cost.  
The system therefore uses a discrete hedge-adjustment framework that seeks a balance between:

- capturing spot variance efficiently,
- avoiding unnecessary turnover,
- and preserving net profitability after frictions.

### 3. Friction-Aware Backtesting

The backtest is explicitly built to move beyond idealized academic returns.  
The research includes realistic implementation frictions such as:

- bid-ask spread,
- execution slippage,
- option transaction costs,
- and spot transaction costs.

This is important because gamma scalping strategies are particularly sensitive to trading frictions.  
A strategy that looks attractive before costs can become much weaker once realistic execution assumptions are imposed.

---

## Performance Summary

The historical backtest showed a **strong absolute return profile**, but also a meaningful trade-off between profitability and portfolio volatility.

### Return Statistics

- Initial Capital: **10,000,000 JPY**
- Final Equity: **21,982,961 JPY**
- Total Return: **119.83%**
- CAGR: **20.58%**
- Total Trades: **79**
- Win Rate: **82.28%**
- Profit Factor: **21.00**
- Average Win: **254,188 JPY**
- Average Loss: **-56,194 JPY**
- Win/Loss Ratio: **4.52**

These statistics suggest a highly asymmetric payoff structure.  
The strategy does not simply rely on frequent small gains; instead, it combines a high win rate with significantly larger average gains than losses, indicating that profitable volatility events were captured more effectively than losing episodes.

The high profit factor is especially notable, as it suggests that when the strategy’s conditions align with realized market movement and volatility behavior, the payoff can be substantial relative to the size of losing trades.

---

## Risk Profile

Although the absolute return was strong, the backtest also revealed that the strategy carries considerable portfolio volatility.

### Risk Metrics

- Sharpe Ratio: **0.35**
- Annualized Volatility: **58.11%**
- Maximum Drawdown (MDD): **-15.95%**
- Calmar Ratio: **1.29**

This means the strategy was profitable, but **not smooth**.  
The equity curve reflects a pattern often seen in long-gamma / long-vega structures: periods of slow or noisy capital progression, interrupted by stronger bursts of profitability when market conditions become favorable.

The low Sharpe Ratio shows that return efficiency per unit of risk still has room for improvement.  
In other words, the strategy appears to contain genuine edge, but the current implementation is still too volatile to be considered fully optimized from an institutional portfolio construction perspective.

---

## Greeks Interpretation

A major part of this research was understanding how the portfolio’s option sensitivities shaped both returns and risk.

### Average Daily Per-Trade Greeks

- Gamma (Γ): **3,250 USD**
- Theta (Θ): **-18,200 JPY**
- Vega (V): **75,400 JPY per 1% IV shift**

These three Greeks explain the economic engine of the strategy:

### Gamma (Γ)
Positive gamma is the core profit engine of the strategy.  
As USD/JPY spot moves, the net delta of the options position changes, creating opportunities to rebalance spot exposure and harvest realized variance.  
This is what allows the strategy to “buy low / sell high” in a mechanical way through dynamic hedging rather than discretionary forecasting.

### Theta (Θ)
Theta represents the daily cost of holding long optionality.  
This is the structural drag that the strategy must overcome.  
During stagnant or low-realized-volatility environments, theta decay can gradually erode portfolio value, which helps explain flatter periods in the cumulative P&L path and contributes to the strategy’s unstable short-term return profile.

### Vega (V)
Vega exposure shows that the portfolio is highly sensitive to changes in implied volatility.  
This is an important source of upside when volatility expands, but it also creates vulnerability when implied volatility compresses.  
As a result, the strategy’s profitability is not purely about spot movement; it is also closely tied to whether volatility repricing occurs in the expected direction.

Taken together, the Greeks profile shows that this is **not a simple trend-following system**.  
It is a structurally long-convexity strategy whose return distribution emerges from the interaction between gamma monetization, theta drag, and vega exposure.

---

## Interpretation of the Payoff Structure

One of the most important findings from the backtest is that the strategy’s returns are **asymmetric but fragile**.

Why asymmetric:
- Profitable trades tend to be meaningfully larger than losing trades.
- Dynamic hedging allows realized spot variance to be monetized.
- Volatility repricing can create a second source of profit beyond spot hedging gains.

Why fragile:
- Theta continuously bleeds when the market remains quiet.
- Excessive hedging can destroy edge through cost accumulation.
- Strong historical performance does not automatically imply stable risk-adjusted performance.

This combination explains why the strategy can look powerful on cumulative return metrics while still showing a modest Sharpe Ratio.

---

## Figures

## Figure 1. USD/JPY Spot, Trade Signals, and Cumulative Net P&L

![USDJPY Signals and PnL](./images/usdjpy_signals.png)

**What this figure shows:**  
The top panel overlays the USD/JPY spot path with strategy trade signals, illustrating when long-volatility entries and exits were triggered across different market conditions.  
The bottom panel displays cumulative **net** P&L after implementation costs, which is important because it reflects a friction-aware performance profile rather than an idealized theoretical return stream.

**Why it matters:**  
This figure summarizes the strategy at the highest level: signal generation, execution activity, and final monetization path.  
It is also the most accessible chart for non-technical readers, since it visually connects market movement with realized strategy outcomes.

### Key interpretation
- Trade entries and exits are distributed across multiple regimes rather than concentrated in a single isolated event.
- The cumulative P&L path trends upward over the test period, suggesting that the strategy was able to convert repeated volatility opportunities into realized profits.
- The fact that the chart is shown **after costs** reinforces that the reported results already embed a more realistic execution environment.

---

## Figure 2. Portfolio Greeks Exposure

![USDJPY Greeks](./images/usdjpy_risk.png)

**What this figure shows:**  
This figure visualizes the strategy’s average per-trade exposure to the three most important option sensitivities: **Gamma**, **Theta**, and **Vega**.  
Instead of treating Greeks as abstract textbook quantities, the chart helps connect them directly to the strategy’s realized behavior over time.

**Why it matters:**  
The Greeks explain the internal mechanics of the return process:

- **Gamma** supports profit generation through re-hedging as spot moves.
- **Theta** represents the daily carrying cost of long optionality and explains why the strategy can bleed in stagnant environments.
- **Vega** captures sensitivity to implied volatility changes and helps explain why the strategy may perform especially well during volatility expansion.

### Key interpretation
- Persistent positive gamma is necessary for the dynamic hedging engine to function as a source of realized trading gains.
- Persistently negative theta highlights that the strategy must keep harvesting enough movement or repricing to justify the cost of long optionality.
- Strong vega exposure reinforces that volatility regime matters: favorable volatility shocks can accelerate gains, while volatility compression can weaken performance.

---

## Limitations and Future Improvements

Although the strategy produced attractive cumulative returns, the research also identified several areas for refinement:

- Improve position sizing so portfolio risk scales more adaptively across volatility regimes.
- Refine delta-hedging logic to better balance turnover and variance capture.
- Add time-based or regime-based filters to reduce theta bleed during less favorable conditions.
- Further distinguish between volatility dislocation opportunities that are structural versus those that are merely noisy short-term deviations.

In future versions, the focus would be less on maximizing raw return and more on improving **risk-adjusted consistency**, especially through better volatility targeting and more adaptive hedge management.

---

## Repository Notes

This repository intentionally omits certain implementation details, exact parameters, and execution thresholds.  
The purpose of this project page is to present the **research framework, portfolio logic, and empirical findings** without disclosing all proprietary strategy settings.

---

## Disclaimer

This project is provided for **research and portfolio demonstration purposes only**.  
It does not constitute investment advice, live trading instruction, or a recommendation to buy or sell any financial instrument.
