# Source Modules (`/src`)

This directory contains the modularized logic of the Deep Hedging system.

### 1. `model.py`

Defines the `LSTMHedger` model.

The architecture consists of:

* A single `LSTMCell` to model sequential market dynamics.
* A Dense output layer with `tanh` activation to generate a hedge ratio (`Δ`) constrained between `-1` and `1`.

At each timestep, the model receives:

* Normalized underlying price (`S_t / K`)
* Black–Scholes delta
* Previous hedge ratio (`Δ_{t-1}`)

### 2. `quant_lib.py`

The quantitative engine containing the mathematical core of the system.

Implemented components include:

* **Black–Scholes Pricing & Greeks**: price, delta, gamma, and theta calculations.
* **Implied Volatility Solver**: bisection-based estimation from observed market prices.
* **PnL Engine**: computes hedging P&L while accounting for transaction costs.
* **CVaR Loss Function**: tail-risk optimization objective used during fine-tuning.

### 3. `data_handler.py`

Handles market data ingestion and preprocessing.

Responsibilities include:

* Filtering raw CSV data for a selected stock/index.
* Separating futures and options contracts.
* Parsing option symbols to extract strike price and option type (Call/Put).
* Computing historical annualized volatility (`σ`) from futures price movements.

### 4. `trainer.py`

Contains the two-phase training pipeline for deep hedging.

#### Phase 1: `run_mse_bootstrapping`

Trains the model to imitate Black–Scholes hedge ratios using Mean Squared Error (MSE).

#### Phase 2: `run_cvar_tuning`

Fine-tunes the model to optimize Conditional Value at Risk (CVaR), enabling more robust hedging under transaction costs and market frictions.
