# Model Checkpoints (`/checkpoints`)

This directory stores the trained neural network weights for the Deep Hedging system.

To prevent conflicts between different financial instruments, model weights are organized in a hierarchical directory structure, where each stock or asset has its own dedicated folder.

## Structure

Each stock is assigned its own sub-directory:

```text
checkpoints/
├── NIFTY/
│   └── model_latest.h5
├── APPLE/
│   └── model_latest.h5
└── RELIANCE/
    └── model_latest.h5
```

## Storage Logic

### Creation

When training a model from scratch on a new stock, the system automatically creates a corresponding sub-directory inside `/checkpoints`.

### Naming Convention

By default, `main.py` stores the most recently trained weights as:

```text
model_latest.h5
```

### Usage

When running the system in **tune** or **predict** mode, the desired checkpoint must be explicitly provided using the `--weights` flag.

Example:

```bash
python main.py --mode predict --stock NIFTY --weights checkpoints/NIFTY/model_latest.h5
```

## Why Separate Folders?

Financial instruments exhibit distinct market dynamics, including:

* Volatility clustering
* Drift behavior
* Mean reversion patterns
* Liquidity and microstructure effects

A model trained on **NIFTY** will generally not hedge **APPLE** or **RELIANCE** effectively.

Maintaining separate checkpoints ensures that the LSTM model remains calibrated to the statistical behavior and risk characteristics of each underlying asset, improving hedging accuracy and robustness.
