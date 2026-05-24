import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from core.ml_oracle import MLRegimeOracle

class TestMLOracle:
    @patch('core.ml_oracle.joblib.load')
    @patch('core.ml_oracle.os.path.exists')
    def test_oracle_initialization(self, mock_exists, mock_load):
        # Setup mock
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Test default path resolution
        oracle = MLRegimeOracle()
        assert oracle.model is not None
        assert "regime_hmm.pkl" in oracle.model_path
        
        # Test custom path
        oracle_custom = MLRegimeOracle(model_path="custom_path.pkl")
        assert oracle_custom.model_path == "custom_path.pkl"
        
    @patch('core.ml_oracle.os.path.exists')
    def test_missing_model_raises_error(self, mock_exists):
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            MLRegimeOracle(model_path="nonexistent.pkl")
            
    @patch('core.ml_oracle.joblib.load')
    @patch('core.ml_oracle.os.path.exists')
    def test_predict_regime_insufficient_data(self, mock_exists, mock_load):
        mock_exists.return_value = True
        
        oracle = MLRegimeOracle()
        
        # Pass fewer than 22 days of data
        short_spread = pd.Series(np.random.randn(20))
        
        # Should fallback to NORMAL without calling the model
        regime = oracle.predict_regime(short_spread)
        assert regime == MLRegimeOracle.REGIME_NORMAL
        oracle.model.predict.assert_not_called()
        
    @patch('core.ml_oracle.joblib.load')
    @patch('core.ml_oracle.os.path.exists')
    def test_predict_regime_valid_data(self, mock_exists, mock_load):
        mock_exists.return_value = True
        mock_model = MagicMock()
        # Mock predict to return a sequence ending in CRISIS (1)
        mock_model.predict.return_value = np.array([0, 0, 1])
        mock_load.return_value = mock_model
        
        oracle = MLRegimeOracle()
        
        # Pass 30 days of valid data
        valid_spread = pd.Series(np.random.randn(30))
        
        regime = oracle.predict_regime(valid_spread)
        
        # The oracle should have called predict on the (N-21) valid rows
        assert oracle.model.predict.called
        assert regime == MLRegimeOracle.REGIME_CRISIS
