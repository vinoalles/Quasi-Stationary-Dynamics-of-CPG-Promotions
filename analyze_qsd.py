#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_qsd.ipynb  (Python version for reproducibility)
Author : Vinodh Kumar Gunasekaran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("promo_data.csv")
print("Loaded:", df.shape, "rows")
print(df.head())

# ------------------------------------------------------------
# 2. BUILD TRANSITION MATRIX Q0
# ------------------------------------------------------------
states = ['TPR', 'Feature', 'Display', 'Feature+Display']
idx_map = {s: i for i, s in enumerate(states)}
n = len(states)
transitions = np.zeros((n, n))

for _, grp in df.groupby(['Retailer', 'UPC']):
    seq = grp['PromoState'].values
    for s1, s2 in zip(seq[:-1], seq[1:]):
        if s1 in states and s2 in states:
            transitions[idx_map[s1], idx_map[s2]] += 1

Q0 = transitions / transitions.sum(axis=1, keepdims=True)
Q0 = np.nan_to_num(Q0)
Q0_df = pd.DataFrame(Q0, index=states, columns=states).round(3)
print("\nTransition Matrix Q0:")
print(Q0_df)

# ------------------------------------------------------------
# 3. COMPUTE m AND α
# ------------------------------------------------------------
eigvals, eigvecs = np.linalg.eig(Q0.T)
alpha_idx = np.argmax(np.real(eigvals))
alpha = 1 - np.real(eigvals[alpha_idx])
m = np.real(eigvecs[:, alpha_idx])
m = m / np.sum(m)

print("\nQuasi-Stationary Distribution (m):")
for s, val in zip(states, m):
    print(f"{s:<15s}: {val:.3f}")
print(f"\nDecay rate α: {alpha:.3f}")
print(f"Expected lifespan (1/α): {1/alpha:.2f} weeks")

# ------------------------------------------------------------
# 4. PLOTS
# ------------------------------------------------------------
# Heatmap of Q0
plt.figure(figsize=(6,5))
sns.heatmap(Q0_df, annot=True, cmap="coolwarm", cbar=True)
plt.title("Transition Probability Heatmap among Active Promotion States")
plt.tight_layout()
plt.savefig("transition_heatmap.png", dpi=300)
plt.show()

# Simulated survival curve (exponential fit)
t = np.arange(0, 20)
S = np.exp(-alpha * t)
plt.figure(figsize=(6,4))
plt.plot(t, S, color='teal', linewidth=3)
plt.title("Promotion Survival Function S(t)=e^{-αt}")
plt.xlabel("Weeks (t)")
plt.ylabel("Probability Promotion Active S(t)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("promotion_survival_curve.png", dpi=300)
plt.show()
