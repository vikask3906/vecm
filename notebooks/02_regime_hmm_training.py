#!/usr/bin/env python3
"""
02_regime_hmm_training.py — ML Oracle Offline Training (Machine Learning Domain)

This script trains a Gaussian Hidden Markov Model (HMM) on historical
cointegrating spread data to classify the market into two latent states:
    - Regime 0: Mean-Reverting (Normal, low volatility, tight spread)
    - Regime 1: Crisis (Unpredictable, high volatility, spread blowout)

Domain: Machine Learning
Strategy/Math: NONE (Uses pre-computed spread)
Dev: Saving a .pkl file

Run:
    python notebooks/02_regime_hmm_training.py
"""

import sys
import os
import warnings

# Suppress hmmlearn deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.data_loader import fetch_prices
from core.johansen import run_johansen
from core.vecm import fit_vecm
from core.spread import construct_spread
import config

def main():
    print(f"\n{'━'*60}")
    print("  STEP 1: Generating Historical Spread for Training")
    print(f"{'━'*60}")
    
    start_date = "2018-01-01"  # Longer lookback for training ML model
    end_date = "2023-12-31"    # Out-of-sample holdout starts 2024
    tickers = config.ASSETS

    print(f"  Fetching training data ({start_date} to {end_date})...")
    prices = fetch_prices(tickers, start=start_date, end=end_date)
    
    # Get the cointegrating vector
    print("  Estimating VECM to extract beta vector...")
    jr = run_johansen(prices, det_order=config.JOHANSEN_DET_ORDER, k_ar_diff=config.JOHANSEN_K_AR_DIFF)
    vr = fit_vecm(prices, rank=max(1, jr.rank), k_ar_diff=config.JOHANSEN_K_AR_DIFF)
    
    # Construct spread
    spread = construct_spread(prices, vr.beta, vec_idx=0)
    print(f"  ✓ Spread constructed: {len(spread)} observations.")

    print(f"\n{'━'*60}")
    print("  STEP 2: Feature Engineering")
    print(f"{'━'*60}")
    
    # Features for HMM:
    # 1. Absolute Spread (deviation from mean)
    # 2. Rolling Volatility of Spread (21-day)
    
    df = pd.DataFrame({"spread": spread})
    df["abs_spread"] = np.abs(df["spread"] - df["spread"].mean())
    df["volatility"] = df["spread"].diff().rolling(21).std()
    df = df.dropna()
    
    print("  Features computed:")
    print("  1. Absolute deviation from mean")
    print("  2. 21-day rolling volatility of spread differences")
    print(f"  ✓ Usable training rows: {len(df)}")
    
    X = df[["abs_spread", "volatility"]].values

    print(f"\n{'━'*60}")
    print("  STEP 3: Training Gaussian Hidden Markov Model")
    print(f"{'━'*60}")
    
    print("  Fitting 2-State GaussianHMM...")
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # Predict states
    states = model.predict(X)
    df["state"] = states
    
    # Identify which state is "Crisis" vs "Mean-Reverting"
    # The crisis state typically has higher volatility and higher absolute spread
    vol_state_0 = df[df["state"] == 0]["volatility"].mean()
    vol_state_1 = df[df["state"] == 1]["volatility"].mean()
    
    if vol_state_1 > vol_state_0:
        crisis_state = 1
        normal_state = 0
    else:
        crisis_state = 0
        normal_state = 1
        
    print(f"  ✓ Model converged: {model.monitor_.converged}")
    print(f"  State {normal_state}: Mean-Reverting (Avg Vol: {df[df['state'] == normal_state]['volatility'].mean():.4f})")
    print(f"  State {crisis_state}: Crisis/Blowout   (Avg Vol: {df[df['state'] == crisis_state]['volatility'].mean():.4f})")
    
    # Map states so that 0 = Normal, 1 = Crisis
    if crisis_state == 0:
        # Swap states so 1 is always crisis
        new_states = 1 - states
        df["mapped_state"] = new_states
        
        # Need to swap model parameters to match the mapped states
        model.means_ = model.means_[::-1]
        model.covars_ = model.covars_[::-1]
        model.startprob_ = model.startprob_[::-1]
        
        # Swap transition matrix rows and cols
        transmat = model.transmat_
        swapped_transmat = np.array([
            [transmat[1,1], transmat[1,0]],
            [transmat[0,1], transmat[0,0]]
        ])
        model.transmat_ = swapped_transmat
    else:
        df["mapped_state"] = states

    print(f"\n{'━'*60}")
    print("  STEP 4: Saving Model and Diagnostics")
    print(f"{'━'*60}")
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "regime_hmm.pkl")
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to: {model_path}")
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("HMM Regime Classification", fontsize=14)
    
    # Spread colored by regime
    axes[0].plot(df.index, df["spread"], color="gray", alpha=0.5, label="Spread")
    axes[0].scatter(df[df["mapped_state"] == 0].index, df[df["mapped_state"] == 0]["spread"], 
                    color="blue", s=5, alpha=0.5, label="Normal Regime")
    axes[0].scatter(df[df["mapped_state"] == 1].index, df[df["mapped_state"] == 1]["spread"], 
                    color="red", s=5, alpha=0.5, label="Crisis Regime")
    axes[0].set_title("Cointegrating Spread by HMM Regime")
    axes[0].legend()
    
    # Volatility colored by regime
    axes[1].plot(df.index, df["volatility"], color="gray", alpha=0.5)
    axes[1].scatter(df[df["mapped_state"] == 0].index, df[df["mapped_state"] == 0]["volatility"], 
                    color="blue", s=5, alpha=0.5)
    axes[1].scatter(df[df["mapped_state"] == 1].index, df[df["mapped_state"] == 1]["volatility"], 
                    color="red", s=5, alpha=0.5)
    axes[1].set_title("Rolling Volatility by HMM Regime")
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "02_hmm_diagnostic.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Diagnostic plot saved to: {plot_path}")
    print(f"\n  Phase 5 ML Offline Training Complete!\n")

if __name__ == "__main__":
    main()
