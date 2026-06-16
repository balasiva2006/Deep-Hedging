# Research Notebooks (`/notebooks`)

This directory contains the exploratory and experimental Jupyter notebooks used during the development of the Deep Hedging system.

These notebooks document the research process, feature engineering, model experimentation, and validation steps that ultimately informed the modularized implementation in `/src`.

## Notebook Overview

### `Delta_Hedging_With_NN.ipynb`

The **initial research notebook** for the project.

This notebook establishes the first end-to-end implementation of the neural-network-based delta hedging framework.

Key components include:

* Market data ingestion and filtering
* Futures and options preprocessing
* **Black–Scholes pricing and Greeks**
* **Implied Volatility (IV) estimation**
* Delta hedging logic
* Initial neural network training experiments
* Baseline performance evaluation

This notebook serves as the foundation of the project and was later refactored into reusable modules inside `/src`.

---

### `Delta_Hedging_With_NN_2.ipynb`

The **second iteration and debugging notebook**.

After observing weak learning behavior in the initial implementation, this notebook focuses on improving training stability and diagnosing model issues.

Key improvements and experiments include:

* Refinements to data preprocessing
* Model learning diagnostics
* Additional visualization and analysis
* Re-evaluation of pricing and hedge signals
* Iterative experimentation to improve convergence

This notebook documents the transition from proof-of-concept experimentation toward a more robust training pipeline.

## Purpose of the `/notebooks` Folder

The notebooks are intentionally kept separate from the production codebase.

They serve three purposes:

1. **Research Documentation**
   Preserve the experimental workflow and reasoning behind architectural choices.

2. **Rapid Prototyping**
   Enable quick testing of new hedging ideas, pricing assumptions, and training strategies.

3. **Reproducibility**
   Allow researchers and contributors to reproduce intermediate experiments before using the modular `/src` implementation.

## Recommended Workflow

For experimentation and model development:

```text
notebooks/ → research & prototyping
```

For production training and inference:

```text
src/ + main.py → modular pipeline
```

The notebooks should be viewed as the **research layer**, while `/src` contains the cleaned and reusable implementation of the finalized system.
