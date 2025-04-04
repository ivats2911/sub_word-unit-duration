#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Models for subword unit duration prediction.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for duration models."""
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class BaselineModel(BaseModel):
    def __init__(self, smooth_factor=1.0):
        self.smooth_factor = smooth_factor
        self.phone_stats = {}
        self.global_mean = 0.0
        self.global_std = 0.0

    def train(self, X_train, y_train, X_val=None, y_val=None):
        from scipy.sparse import issparse

        num_phones = getattr(X_train, 'shape', [])[1] if hasattr(X_train, 'shape') else 0
        if hasattr(X_train, 'num_phones'):
            num_phones = X_train.num_phones

        if issparse(X_train):
            phone_indices = X_train[:, :num_phones].argmax(axis=1).A1
        else:
            phone_indices = X_train[:, :num_phones].argmax(axis=1)

        df = pd.DataFrame({'phone': phone_indices, 'duration': y_train})

        self.global_mean = np.mean(y_train)
        self.global_std = np.std(y_train)

        phone_stats = df.groupby('phone').agg({
            'duration': ['count', 'mean', 'std']
        })
        phone_stats.columns = ['_'.join(col).strip() for col in phone_stats.columns.values]
        phone_stats = phone_stats.reset_index()

        for idx, row in phone_stats.iterrows():
            count = row['duration_count']
            mean = row['duration_mean']
            std = row['duration_std'] if not np.isnan(row['duration_std']) else 0.0

            smoothed_mean = (count * mean + self.smooth_factor * self.global_mean) / (count + self.smooth_factor)
            smoothed_std = np.sqrt((count * std**2 + self.smooth_factor * self.global_std**2) / (count + self.smooth_factor))

            self.phone_stats[row['phone']] = {
                'mean': smoothed_mean,
                'std': smoothed_std,
                'count': count
            }

        logger.info(f"Trained baseline model with {len(self.phone_stats)} phones")
        return self

    def predict(self, X):
        from scipy.sparse import issparse

        num_phones = getattr(X, 'shape', [])[1] if hasattr(X, 'shape') else 0
        if hasattr(X, 'num_phones'):
            num_phones = X.num_phones

        if issparse(X):
            phone_indices = X[:, :num_phones].argmax(axis=1).A1
        else:
            phone_indices = X[:, :num_phones].argmax(axis=1)

        predictions = []
        for idx in phone_indices:
            if idx in self.phone_stats:
                predictions.append(self.phone_stats[idx]['mean'])
            else:
                predictions.append(self.global_mean)

        return np.array(predictions)




class LinearModel(BaseModel):
    """Linear regression model for duration prediction."""
    
    def __init__(self):
        """Initialize the linear model."""
        self.model = LinearRegression()
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the linear model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (not used)
            y_val: Validation targets (not used)
        """
        logger.info("Training linear regression model")
        self.model.fit(X_train, y_train)
        
        # Log training results
        train_pred = self.model.predict(X_train)
        train_error = np.mean(np.abs(train_pred - y_train))
        logger.info(f"Training MAE: {train_error:.4f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_error = np.mean(np.abs(val_pred - y_val))
            logger.info(f"Validation MAE: {val_error:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the linear model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted durations
        """
        return self.model.predict(X)


class TreeModel(BaseModel):
    """Tree-based model (Random Forest or XGBoost) for duration prediction."""
    
    def __init__(self, model_type='rf', **kwargs):
        """
        Initialize the tree model.
        
        Args:
            model_type: Type of tree model ('rf' or 'xgboost')
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        
        if model_type == 'rf':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the tree model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info(f"Training {self.model_type} model")
        
        if X_val is not None and y_val is not None and hasattr(self.model, 'fit_with_eval_set'):
            # For XGBoost, use validation set for early stopping
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=False)
        else:
            self.model.fit(X_train, y_train)
        
        # Log training results
        train_pred = self.model.predict(X_train)
        train_error = np.mean(np.abs(train_pred - y_train))
        logger.info(f"Training MAE: {train_error:.4f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_error = np.mean(np.abs(val_pred - y_val))
            logger.info(f"Validation MAE: {val_error:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the tree model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted durations
        """
        return self.model.predict(X)


class LSTMModel(BaseModel):
    """LSTM-based neural network for duration prediction."""
    
    class LSTM(nn.Module):
        """LSTM neural network architecture."""
        
        def __init__(self, input_size, hidden_size, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )
        
        def forward(self, x):
            # Reshape input to (batch, seq_len, features) if needed
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])  # Use last time step
            return output.squeeze(-1)
    
    def __init__(self, input_size, hidden_size=128, dropout=0.2, batch_size=32, epochs=50, lr=0.001):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            dropout: Dropout rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr: Learning rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        
        # Initialize model, loss, and optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Will be initialized during training when input size is known
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info(f"Training LSTM model on {self.device}")
        
        # Initialize model
        self.model = self.LSTM(
            input_size=X_train.shape[1],
            hidden_size=self.hidden_size,
            dropout=self.dropout
        ).to(self.device)
        
        # Create data loaders
        X_train_tensor = torch.FloatTensor(X_train.toarray()).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val.toarray()).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the LSTM model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted durations
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X.toarray()).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions


def train_model(X_train, y_train, X_val, y_val, model_type='baseline', config=None):
    """
    Train a model for duration prediction.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_type: Type of model to train
        config: Configuration object
        
    Returns:
        Trained model
    """
    if model_type == 'baseline':
        model = BaselineModel(smooth_factor=config.smooth_factor)
    elif model_type == 'linear':
        model = LinearModel()
    elif model_type in ['rf', 'random_forest']:
        model = TreeModel(
            model_type='rf',
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_seed
        )
    elif model_type == 'xgboost':
        model = TreeModel(
            model_type='xgboost',
            n_estimators=config.n_estimators,
            max_depth=config.max_depth if config.max_depth is not None else 6,
            learning_rate=config.learning_rate,
            random_state=config.random_seed
        )
    elif model_type == 'lstm':
        model = LSTMModel(
            input_size=X_train.shape[1],
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            batch_size=config.batch_size,
            epochs=config.epochs,
            lr=config.learning_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model.train(X_train, y_train, X_val, y_val)
    
    return model


def load_model(path):
    """
    Load a model from disk.
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded model
    """
    return BaseModel.load(path)