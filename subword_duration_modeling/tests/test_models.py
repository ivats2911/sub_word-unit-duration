#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for model implementations.
"""

import os
import sys
import unittest
import numpy as np
from sklearn.datasets import make_regression
from scipy.sparse import csr_matrix

# Add project root to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import BaselineModel, LinearModel, TreeModel


class TestModels(unittest.TestCase):
    """Tests for model implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data for testing
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
        
        # Number of phones/phoneme features to simulate
        num_phones = 10
        
        # Create one-hot encoding for phone identities
        X_phones = np.zeros((X.shape[0], num_phones))
        phone_ids = np.clip(np.floor((X[:, 0] + 3) / 6 * num_phones), 0, num_phones-1).astype(int)
        for i, p_id in enumerate(phone_ids):
            X_phones[i, p_id] = 1
        
        # Combine phone features with other features
        X_combined = np.hstack([X_phones, X])
        
        # Split into train and test
        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)
        
        self.X_train = X_combined[train_idx]
        self.y_train = y[train_idx]
        self.X_test = X_combined[test_idx]
        self.y_test = y[test_idx]
        
        # Create sparse matrix versions
        self.X_train_sparse = csr_matrix(self.X_train)
        self.X_test_sparse = csr_matrix(self.X_test)
        
        # Store the number of phones for reference
        self.num_phones = num_phones
        self.X_train_sparse.num_phones = num_phones
        self.X_test_sparse.num_phones = num_phones

    def test_baseline_model(self):
        """Test baseline model with both dense and sparse inputs."""
        # Test with dense data
        model_dense = BaselineModel(smooth_factor=1.0)
        model_dense.train(self.X_train, self.y_train)
        
        # Check that phone statistics are calculated
        self.assertGreater(len(model_dense.phone_stats), 0)
        
        # Make predictions
        predictions_dense = model_dense.predict(self.X_test)
        
        # Check that predictions have the right shape
        self.assertEqual(len(predictions_dense), len(self.X_test))
        
        # Check that predictions are reasonable (not all zeros or NaN)
        self.assertFalse(np.all(predictions_dense == 0))
        self.assertFalse(np.any(np.isnan(predictions_dense)))
        
        # Test with sparse data
        model_sparse = BaselineModel(smooth_factor=1.0)
        model_sparse.train(self.X_train_sparse, self.y_train)
        
        # Make predictions
        predictions_sparse = model_sparse.predict(self.X_test_sparse)
        
        # Check that predictions have the right shape
        self.assertEqual(len(predictions_sparse), self.X_test_sparse.shape[0])
        
        # Check that predictions are reasonable (not all zeros or NaN)
        self.assertFalse(np.all(predictions_sparse == 0))
        self.assertFalse(np.any(np.isnan(predictions_sparse)))

    def test_linear_model(self):
        """Test linear regression model."""
        # Initialize and train the model
        model = LinearModel()
        model.train(self.X_train, self.y_train)
        
        # Make predictions
        predictions = model.predict(self.X_test)
        
        # Check that predictions have the right shape
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check that predictions are reasonable (not all zeros or NaN)
        self.assertFalse(np.all(predictions == 0))
        self.assertFalse(np.any(np.isnan(predictions)))

    def test_tree_model_rf(self):
        """Test Random Forest model."""
        # Initialize and train the model
        model = TreeModel(model_type='rf', n_estimators=10, random_state=42)
        model.train(self.X_train, self.y_train)
        
        # Make predictions
        predictions = model.predict(self.X_test)
        
        # Check that predictions have the right shape
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check that predictions are reasonable (not all zeros or NaN)
        self.assertFalse(np.all(predictions == 0))
        self.assertFalse(np.any(np.isnan(predictions)))

    def test_tree_model_xgboost(self):
        """Test XGBoost model."""
        # Initialize and train the model
        model = TreeModel(model_type='xgboost', n_estimators=10, random_state=42)
        model.train(self.X_train, self.y_train)
        
        # Make predictions
        predictions = model.predict(self.X_test)
        
        # Check that predictions have the right shape
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check that predictions are reasonable (not all zeros or NaN)
        self.assertFalse(np.all(predictions == 0))
        self.assertFalse(np.any(np.isnan(predictions)))

    def test_model_save_load(self):
        """Test saving and loading models."""
        # Train a model
        model = LinearModel()
        model.train(self.X_train, self.y_train)
        
        # Save the model
        os.makedirs('test_models', exist_ok=True)
        save_path = 'test_models/test_model.pkl'
        model.save(save_path)
        
        # Load the model
        loaded_model = LinearModel.load(save_path)
        
        # Check that predictions are the same
        orig_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
        
        # Clean up
        os.remove(save_path)
        os.rmdir('test_models')


if __name__ == '__main__':
    unittest.main()