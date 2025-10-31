# Quasi-Stationary Dynamics of CPG Promotions

This repository accompanies the paper:

> Gunasekaran, V. K. (2025). *Quasi-Stationary Dynamics of CPG Promotions*.

It provides reproducible Python code and simulated data illustrating how 
promotion persistence, decay rate (α), and conditional composition (m) 
are estimated in consumer-packaged-goods analytics.

## Files
- **simulate_promo_data.py** — creates `promo_data.csv` (~10 000 rows)
- **analyze_qsd.ipynb** — computes transition matrix Q⁽⁰⁾, eigen decomposition, and plots survival/heatmap
- **promo_data.csv** — generated dataset (if large, provide link or ZIP)
- **transition_heatmap.png**, **promotion_survival_curve.png** — output figures

## Usage
```bash
python simulate_promo_data.py
python analyze_qsd.ipynb   # or open in Jupyter / VS Code
