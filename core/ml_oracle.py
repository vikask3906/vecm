# core/ml_oracle.py
"""
Machine Learning Oracle for Market Regime Classification.

This module loads the offline-trained Hidden Markov Model (HMM) and provides
live inference capabilities for Phase 5. The Oracle determines whether the
current market is in a "Mean-Reverting" (Normal) regime or a "Crisis" (Blowout)
regime based on recent spread volatility and magnitude.
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress warnings from joblib/sklearn during live inference
warnings.filterwarnings("ignore", category=UserWarning)

class MLRegimeOracle:
    """
    Live inference engine for the HMM Regime Classifier.
    """
    
    REGIME_NORMAL = 0
    REGIME_CRISIS = 1
    
    def __init__(self, model_path: str = None):
        """
        Initialize the Oracle by loading the serialized HMM.
        
        Parameters
        ----------
        model_path : str, optional
            Path to the .pkl model. Defaults to 'models/regime_hmm.pkl' in project root.
        """
        if model_path is None:
            # Resolve default path relative to this file's location
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "regime_hmm.pkl")
            
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ML Oracle missing model file: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        
    def predict_regime(self, recent_spreads: pd.Series) -> int:
        """
        Predict the current market regime based on a window of recent spreads.
        
        Parameters
        ----------
        recent_spreads : pd.Series
            The last N days of the cointegrating spread (needs at least 22 days 
            to compute a 21-day rolling volatility).
            
        Returns
        -------
        int
            0 for REGIME_NORMAL (Mean-Reverting)
            1 for REGIME_CRISIS (Blowout/Unpredictable)
        """
        if len(recent_spreads) < 22:
            # Fallback to normal if insufficient data
            return self.REGIME_NORMAL
            
        # Feature 1: Absolute Spread (centered around mean)
        abs_spread = np.abs(recent_spreads - recent_spreads.mean())
        
        # Feature 2: 21-day rolling volatility of the differences
        diffs = recent_spreads.diff()
        volatility = diffs.rolling(21).std()
        
        # We only need the features for the very last observation (today)
        # But HMMs predict sequences, so we predict on the available valid sequence
        # and take the last state.
        
        # Construct DataFrame to drop NaNs properly
        df = pd.DataFrame({
            "abs_spread": abs_spread,
            "volatility": volatility
        }).dropna()
        
        if len(df) == 0:
            return self.REGIME_NORMAL
            
        X = df.values
        
        # Predict the sequence of states
        states = self.model.predict(X)
        
        # The current regime is the last predicted state
        current_regime = states[-1]
        
        return current_regime
