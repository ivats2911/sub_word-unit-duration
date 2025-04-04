#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering for subword unit duration modeling.
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack
import time

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, config: Any):
        self.config = config
        self.phone_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.class_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.fitted = False
        self.num_phones = 0

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(['utterance_id', 'phone_start'])
        context_size = self.config.context_size

        for i in range(1, context_size + 1):
            df[f'prev_{i}_phone'] = 'PAD'
            df[f'next_{i}_phone'] = 'PAD'
            df[f'prev_{i}_class'] = 'PAD'
            df[f'next_{i}_class'] = 'PAD'

        for i in range(1, context_size + 1):
            df[f'prev_{i}_phone'] = df.groupby('utterance_id')['phone'].shift(i).fillna('PAD')
            df[f'prev_{i}_class'] = df.groupby('utterance_id')['phone_class'].shift(i).fillna('PAD')
            df[f'next_{i}_phone'] = df.groupby('utterance_id')['phone'].shift(-i).fillna('PAD')
            df[f'next_{i}_class'] = df.groupby('utterance_id')['phone_class'].shift(-i).fillna('PAD')

        return df

    def fit(self, df: pd.DataFrame) -> None:
        logger.info("Fitting feature extractors")
        self.phone_encoder.fit(df['phone'].fillna('PAD').values.reshape(-1, 1))
        self.class_encoder.fit(df['phone_class'].fillna('PAD').values.reshape(-1, 1))
        self.num_phones = len(self.phone_encoder.categories_[0])
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> Tuple[Any, np.ndarray]:
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        logger.info("Transforming data into features")
        df = self._add_context_features(df)
        feature_blocks = []

        # 1. Phone identity
        phone_onehot = self.phone_encoder.transform(df['phone'].fillna('PAD').values.reshape(-1, 1))
        feature_blocks.append(phone_onehot)

        # 2. Phone class
        if self.config.use_phoneme_class:
            class_onehot = self.class_encoder.transform(df['phone_class'].fillna('PAD').values.reshape(-1, 1))
            feature_blocks.append(class_onehot)

        # 3. Context features
        for i in range(1, self.config.context_size + 1):
            prev_phone_vals = df[f'prev_{i}_phone'].fillna('PAD').values.reshape(-1, 1)
            next_phone_vals = df[f'next_{i}_phone'].fillna('PAD').values.reshape(-1, 1)
            prev_phone = self.phone_encoder.transform(prev_phone_vals)
            next_phone = self.phone_encoder.transform(next_phone_vals)
            feature_blocks.extend([prev_phone, next_phone])

            if self.config.use_phoneme_class:
                prev_class_vals = df[f'prev_{i}_class'].fillna('PAD').values.reshape(-1, 1)
                next_class_vals = df[f'next_{i}_class'].fillna('PAD').values.reshape(-1, 1)
                prev_class = self.class_encoder.transform(prev_class_vals)
                next_class = self.class_encoder.transform(next_class_vals)
                feature_blocks.extend([prev_class, next_class])

        # 4. Position features
        if self.config.use_word_position:
            feature_blocks.append(df[['phone_pos_in_word']].values)
        if self.config.use_sentence_position:
            feature_blocks.append(df[['word_pos_in_utterance']].values)

        # 5. Speaking rate
        if self.config.use_speaking_rate:
            feature_blocks.append(df[['speaking_rate']].values)

        X = hstack(feature_blocks).tocsr()
        y = df['phone_duration'].values

        return X, y

def extract_features(df: pd.DataFrame, config: Any, extractor: FeatureExtractor = None, max_samples: int = None) -> Tuple[Any, np.ndarray, FeatureExtractor]:
    if extractor is None:
        extractor = FeatureExtractor(config)
        extractor.fit(df)
        config.num_phones = extractor.num_phones  # Store this for downstream models

    if max_samples is not None:
        df = df.iloc[:max_samples]
        logger.info(f"Using only {max_samples} samples for feature extraction.")

    start = time.time()
    X, y = extractor.transform(df)
    logger.info(f"Feature extraction took {time.time() - start:.2f} seconds.")
    return X, y, extractor

