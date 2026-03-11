# 📈 Quantitative Trading & Machine Learning Portfolio

Welcome to my portfolio.

I focus on building **data-driven trading systems**, **financial machine learning models**, and **high-performance backend architectures** for quantitative research and production environments.

My work sits at the intersection of **quantitative finance, machine learning, and system engineering**, combining rigorous data science experimentation with scalable infrastructure.

---

# Core Expertise

- **Algorithmic Trading System Design**
- **Financial Machine Learning**
- **Quantitative Alpha Research**
- **Event-Driven Trading Architecture**
- **High-Performance Data Pipelines**
- **Risk-Aware Model Engineering**

---

# Tech Stack

### Programming Languages
- **Python**
- **Rust**
- **JavaScript**

### Machine Learning
- XGBoost  
- Random Forest  
- Scikit-Learn  
- Reinforcement Learning (PPO)  
- SHAP Model Interpretability  

### Data Engineering
- Polars  
- Pandas  
- Parquet  
- High-Frequency Tick Data Processing  

### Backend & Infrastructure
- RESTful APIs  
- Event-Driven Trading Engines  
- **PostgreSQL**
- **Docker containerization**

### Frontend
- Vue.js (Data Visualization & Dashboard Interfaces)

### Quantitative Finance
- Alpha Factor Engineering  
- Multi-Timeframe Market Analysis  
- Risk Management (CVaR, Cost-Sensitive Learning)  
- Liquidity & Slippage Modeling  

---

# 📂 Featured Projects

## 1️⃣ AI Trading Agent: High-Fidelity Market Simulation & RL Framework

**Domain**  
Quantitative Trading · Reinforcement Learning · Full-Stack System

**Overview**

Developed an **AI trading agent trained in a high-fidelity market simulator** that reproduces realistic Binance market microstructure using an **event-driven architecture**.

The agent uses a **hierarchical decision model**, separating:

- **Macro strategy reasoning (Cognition layer)**
- **Micro trade execution (Reflex layer)**

This design prevents **look-ahead bias** while allowing realistic interaction with simulated market liquidity.

**Key Highlights**

- Built a **high-performance data pipeline** using **Polars + Rust** for fast order-flow synthesis
- Designed a proprietary **Slippage model** to simulate realistic liquidity constraints
- Implemented a **Vue.js trading dashboard** for strategy visualization
- Developed a **Rust backend API** for high-throughput data access

---

## 2️⃣ Multi-Market Defensive Alpha: Machine Learning Factor Research

**Domain**  
Financial Machine Learning · Alpha Research

**Overview**

A cross-asset predictive trading model covering **30+ instruments** including **US equities, ETFs, and cryptocurrencies**.

The system is designed as a **Defensive Alpha strategy**, aiming to **preserve capital during market downturns while maintaining long-term growth**.

**Key Highlights**

- Engineered **20+ structural market features**
- Developed two proprietary quantitative factors:

**VARM**  
Volatility-Adjusted Relative Momentum

**PVD**  
Price-Volume Divergence

- Reduced market noise by using **Random Forest filtering**, outperforming baseline **XGBoost models**
- Performed deep **SHAP analysis** to validate that the model learned:
  - **Low Volatility Anomaly**
  - **Mean Reversion dynamics**

---

## 3️⃣ Credit Risk Prediction: Cost-Sensitive Learning for Imbalanced Data

**Domain**  
Risk Modeling · Classification

**Overview**

A machine learning pipeline designed to predict **credit card default risk** using a dataset of **30,000 clients**.

The project addresses the severe **class imbalance problem** commonly found in financial risk datasets.

**Key Highlights**

- Engineered interpretable financial stress indicators:
  - **Credit Utilization**
  - **Payment Ratios**
- Optimized for **Recall**, prioritizing the detection of potential defaulters
- Implemented **Cost-Sensitive XGBoost**, significantly improving detection of true default cases compared to standard models

---

# Engineering Principles

Across these projects, I focus on:

- **Research reproducibility**
- **Production-ready system design**
- **High-performance data processing**
- **Model interpretability and risk awareness**

The goal is to build **quantitative systems that are not only predictive, but also deployable in real-world trading environments**.

---

# Contact

If you're interested in **quantitative research, trading systems, or financial machine learning collaboration**, feel free to reach out.
