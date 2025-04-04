#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing module for subword unit duration modeling.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load JSON data recursively from all files in the specified directory and its subdirectories.
    
    Args:
        data_dir: Directory containing JSON files directly or in subdirectories
        
    Returns:
        List of parsed JSON objects (dictionaries only)
    """
    logger.info(f"Loading data from {data_dir}")
    data = []
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist")
        return data

    def process_json_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, dict):
                    result = json_data.get("result", {})
                    if isinstance(result, dict):  # ✅ extra protection
                        data.append(json_data)
                    else:
                        logger.warning(f"File {file_path} has non-dict 'result': {type(result)}")
                else:
                    logger.warning(f"File {file_path} is not a dict: {type(json_data)}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")


    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                process_json_file(file_path)

    logger.info(f"Successfully loaded {len(data)} valid JSON files from {data_dir}")
    return data


def extract_phone_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract phone-level data from JSON objects.
    
    Args:
        data: List of parsed JSON objects
        
    Returns:
        DataFrame with phone-level information
    """
    logger.info("Extracting phone-level data")
    phone_data = []

    for utterance_idx, utterance in enumerate(data):
        result = utterance.get('result', {})
        text = result.get('text_normalized', '')

        if not text:
            logger.warning(f"Utterance {utterance_idx} has no normalized text, skipping")
            continue

        segments = result.get("segments", [])
        if not segments:
            logger.warning(f"Utterance {utterance_idx} has no segments, skipping")
            continue

        for segment_idx, segment in enumerate(segments):
            words = segment.get("words", [])
            for word_idx, word in enumerate(words):
                if word.get('oov', False):
                    continue

                word_text = word.get('word_normalized', '')
                word_start = word.get('start', 0.0)
                word_duration = word.get('duration', 0.0)

                word_pos_in_utterance = word_idx / len(words) if words else 0
                phones = word.get('phones', [])

                for phone_idx, phone in enumerate(phones):
                    phone_text = phone.get('phone', '')
                    phone_start = phone.get('start', 0.0)
                    phone_duration = phone.get('duration', 0.0)
                    phone_class = phone.get('class', '')

                    phone_pos_in_word = phone_idx / len(phones) if phones else 0

                    phone_data.append({
                        'utterance_id': f"{utterance_idx}_{segment_idx}",
                        'utterance_text': text,
                        'word': word_text,
                        'word_idx': word_idx,
                        'word_pos_in_utterance': word_pos_in_utterance,
                        'word_start': word_start,
                        'word_duration': word_duration,
                        'phone': phone_text,
                        'phone_idx': phone_idx,
                        'phone_pos_in_word': phone_pos_in_word,
                        'phone_start': phone_start,
                        'phone_duration': phone_duration,
                        'phone_class': phone_class,
                    })

    df = pd.DataFrame(phone_data)
    logger.info(f"Extracted data for {len(df)} phones from {len(data)} utterances")

    return df



def calculate_speaking_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate speaking rate features.
    
    Args:
        df: DataFrame with phone data
        
    Returns:
        DataFrame with added speaking rate features
    """
    # Group by utterance and calculate stats
    utterance_stats = df.groupby('utterance_id').agg({
        'phone_duration': ['mean', 'std', 'sum'],
        'phone': 'count'
    })
    
    utterance_stats.columns = ['_'.join(col).strip() for col in utterance_stats.columns.values]
    utterance_stats.reset_index(inplace=True)
    
    # Calculate speaking rate (phones per second)
    utterance_stats['speaking_rate'] = utterance_stats['phone_count'] / utterance_stats['phone_duration_sum']
    
    # Calculate normalized phone duration (z-score within utterance)
    result = df.merge(utterance_stats[['utterance_id', 'phone_duration_mean', 'phone_duration_std', 'speaking_rate']], 
                      on='utterance_id')
    
    result['phone_duration_norm'] = (result['phone_duration'] - result['phone_duration_mean']) / \
                                   result['phone_duration_std'].replace(0, 1)  # Avoid division by zero
    
    return result


def preprocess_data(native_data: List[Dict[str, Any]], 
                    non_native_data: List[Dict[str, Any]],
                    native_only: bool = False,
                    config: Any = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data and split into train, validation, and test sets.
    
    Args:
        native_data: List of native speaker data
        non_native_data: List of non-native speaker data
        native_only: Whether to use only native speaker data for training
        config: Configuration object
        
    Returns:
        Tuple of (train_data, val_data, test_data) DataFrames
    """
    logger.info("Preprocessing data")
    
    # Extract phone-level data
    native_df = extract_phone_data(native_data)
    native_df['is_native'] = True
    
    non_native_df = extract_phone_data(non_native_data)
    non_native_df['is_native'] = False
    
    # Calculate speaking rate features
    native_df = calculate_speaking_rate(native_df)
    non_native_df = calculate_speaking_rate(non_native_df)
    
    # Determine training, validation and test sets
    if native_only:
        # Split native data into train/val/test
        train_df, temp_df = train_test_split(
            native_df, 
            test_size=(config.val_ratio + config.test_ratio),
            random_state=config.random_seed
        )
        
        val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=(1 - val_ratio_adjusted),
            random_state=config.random_seed
        )
        
        # Add non-native data to test set for evaluation
        test_df = pd.concat([test_df, non_native_df])
    else:
        # Combine all data and split
        all_data = pd.concat([native_df, non_native_df])
        
        train_df, temp_df = train_test_split(
            all_data, 
            test_size=(config.val_ratio + config.test_ratio),
            random_state=config.random_seed,
            stratify=all_data['is_native']  # Stratify by native/non-native
        )
        
        val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=(1 - val_ratio_adjusted),
            random_state=config.random_seed,
            stratify=temp_df['is_native']  # Stratify by native/non-native
        )
    
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def calculate_phone_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each phone.
    
    Args:
        df: DataFrame with phone data
        
    Returns:
        DataFrame with phone statistics
    """
    phone_stats = df.groupby('phone').agg({
        'phone_duration': ['count', 'mean', 'std', 'min', 'max'],
        'phone_duration_norm': ['mean', 'std']
    })
    
    phone_stats.columns = ['_'.join(col).strip() for col in phone_stats.columns.values]
    phone_stats.reset_index(inplace=True)
    
    # Replace NaN std with 0
    phone_stats.fillna(0, inplace=True)
    
    return phone_stats