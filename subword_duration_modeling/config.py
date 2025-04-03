#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration settings for the subword unit duration modeling project.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Config:
    """Configuration for the project."""
    
    # Data settings
    data_dir: str = 'data'
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    native_only: bool = False
    min_instances: int = 5  # Minimum instances of a phoneme to include
    
    # Feature settings
    context_size: int = 2  # Number of phonemes before and after
    use_word_position: bool = True
    use_sentence_position: bool = True
    use_phoneme_class: bool = True
    use_stress: bool = True  # If stress information is available
    use_speaking_rate: bool = True
    
    # Model settings
    model_type: str = 'baseline'  # baseline, linear, rf, xgboost, lstm, transformer
    random_seed: int = 42
    
    # Baseline model settings
    smooth_factor: float = 1.0  # Laplace smoothing
    
    # Machine learning model settings
    cv_folds: int = 5
    
    # Tree-based model settings
    n_estimators: int = 100
    max_depth: int = None
    
    # Neural network settings
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    hidden_size: int = 128
    dropout: float = 0.2
    
    # Evaluation settings
    metrics: List[str] = field(default_factory=lambda: ['mae', 'rmse', 'correlation'])
    
    # Paths
    model_dir: str = 'models'
    output_dir: str = 'reports'
    
    def update_from_args(self, args):
        """Update config from command line arguments."""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {key: getattr(self, key) for key in self.__annotations__}