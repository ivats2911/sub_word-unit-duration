#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering for subword unit duration modeling.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Feature extractor class for subword unit duration modeling."""
    
    def __init__(self, config: Any):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.phone_encoder = LabelEncoder()
        self.class_encoder = OneHotEncoder(sparse=False)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add context features for each phone.
        
        Args:
            df: DataFrame with phone data
            
        Returns:
            DataFrame with added context features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Sort by utterance and start time to ensure correct order
        result = result.sort_values(['utterance_id', 'phone_start'])
        
        # Add context features
        context_size = self.config.context_size
        
        # Initialize columns for previous and next phones
        for i in range(1, context_size + 1):
            result[f'prev_{i}_phone'] = 'PAD'
            result[f'next_{i}_phone'] = 'PAD'
            result[f'prev_{i}_class'] = 'PAD'
            result[f'next_{i}_class'] = 'PAD'
        
        # Group by utterance and shift to get previous and next phones
        for i in range(1, context_size + 1):
            # Previous phones
            result[f'prev_{i}_phone'] = result.groupby('utterance_id')['phone'].shift(i).fillna('PAD')
            result[f'prev_{i}_class'] = result.groupby('utterance_id')['phone_class'].shift(i).fillna('PAD')
            
            # Next phones
            result[f'next_{i}_phone'] = result.groupby('utterance_id')['phone'].shift(-i).fillna('PAD')
            result[f'next_{i}_class'] = result.groupby('utterance_id')['phone_class'].shift(-i).fillna('PAD')
        
        return result
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit feature extractors on training data.
        
        Args:
            df: Training DataFrame
        """
        logger.info("Fitting feature extractors")
        
        # Fit phone encoder
        self.phone_encoder.fit(
            df['phone'].tolist() + 
            ['PAD'] +  # Add padding token
            [f"{p}_{c}" for p in df['phone'].unique() for c in ['initial', 'medial', 'final']]  # Position variants
        )
        
        # Fit class encoder
        self.class_encoder.fit(
            np.array(df['phone_class'].tolist() + ['PAD']).reshape(-1, 1)
        )
        
        # Mark as fitted
        self.fitted = True
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data into features and targets.
        
        Args:
            df: DataFrame with phone data
            
        Returns:
            Tuple of (X, y) arrays
        """
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        logger.info("Transforming data into features")
        
        # Add context features
        df_with_context = self._add_context_features(df)
        
        # Prepare feature lists
        feature_lists = []
        
        # 1. Phone identity features (one-hot encoded)
        phone_encoded = self.phone_encoder.transform(df_with_context['phone'])
        phone_onehot = np.eye(len(self.phone_encoder.classes_))[phone_encoded]
        feature_lists.append(phone_onehot)
        
        # 2. Phone class features (if enabled)
        if self.config.use_phoneme_class:
            class_encoded = self.class_encoder.transform(
                np.array(df_with_context['phone_class']).reshape(-1, 1)
            )
            feature_lists.append(class_encoded)
        
        # 3. Context features (if context_size > 0)
        if self.config.context_size > 0:
            for i in range(1, self.config.context_size + 1):
                # Previous phone context
                prev_phone_encoded = self.phone_encoder.transform(df_with_context[f'prev_{i}_phone'])
                prev_phone_onehot = np.eye(len(self.phone_encoder.classes_))[prev_phone_encoded]
                feature_lists.append(prev_phone_onehot)
                
                # Next phone context
                next_phone_encoded = self.phone_encoder.transform(df_with_context[f'next_{i}_phone'])
                next_phone_onehot = np.eye(len(self.phone_encoder.classes_))[next_phone_encoded]
                feature_lists.append(next_phone_onehot)
                
                # Previous and next phone class (if enabled)
                if self.config.use_phoneme_class:
                    prev_class_encoded = self.class_encoder.transform(
                        np.array(df_with_context[f'prev_{i}_class']).reshape(-1, 1)
                    )
                    feature_lists.append(prev_class_encoded)
                    
                    next_class_encoded = self.class_encoder.transform(
                        np.array(df_with_context[f'next_{i}_class']).reshape(-1, 1)
                    )
                    feature_lists.append(next_class_encoded)
        
        # 4. Position features
        if self.config.use_word_position:
            # Add word position features
            feature_lists.append(df_with_context[['phone_pos_in_word']].values)
        
        if self.config.use_sentence_position:
            # Add sentence position features
            feature_lists.append(df_with_context[['word_pos_in_utterance']].values)
        
        # 5. Speaking rate features
        if self.config.use_speaking_rate:
            feature_lists.append(df_with_context[['speaking_rate']].values)
        
        # Combine all features
        X = np.hstack(feature_lists)
        
        # Target values (phone durations)
        y = df_with_context['phone_duration'].values
        
        return X, y


def extract_features(df: pd.DataFrame, config: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from data.
    
    Args:
        df: DataFrame with phone data
        config: Configuration object
        
    Returns:
        Tuple of (X, y) arrays
    """
    extractor = FeatureExtractor(config)
    extractor.fit(df)
    return extractor.transform(df)