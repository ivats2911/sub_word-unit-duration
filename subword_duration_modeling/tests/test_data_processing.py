#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for data processing module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import extract_phone_data, calculate_speaking_rate


class TestDataProcessing(unittest.TestCase):
    """Tests for data_processing module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test utterance with the structure matching your data
        self.test_utterance = {
            'result': {
                'text_normalized': 'TEST UTTERANCE',
                'segments': [
                    {
                        'words': [
                            {
                                'word_normalized': 'TEST',
                                'start': 0.0,
                                'duration': 0.5,
                                'oov': False,
                                'phones': [
                                    {'phone': 't', 'start': 0.0, 'duration': 0.1, 'class': 'C'},
                                    {'phone': 'e', 'start': 0.1, 'duration': 0.2, 'class': 'V'},
                                    {'phone': 's', 'start': 0.3, 'duration': 0.1, 'class': 'C'},
                                    {'phone': 't', 'start': 0.4, 'duration': 0.1, 'class': 'C'}
                                ]
                            },
                            {
                                'word_normalized': 'UTTERANCE',
                                'start': 0.6,
                                'duration': 0.9,
                                'oov': False,
                                'phones': [
                                    {'phone': 'u', 'start': 0.6, 'duration': 0.1, 'class': 'V'},
                                    {'phone': 't', 'start': 0.7, 'duration': 0.1, 'class': 'C'},
                                    {'phone': 't', 'start': 0.8, 'duration': 0.1, 'class': 'C'},
                                    {'phone': 'e', 'start': 0.9, 'duration': 0.2, 'class': 'V'},
                                    {'phone': 'r', 'start': 1.1, 'duration': 0.1, 'class': 'C'},
                                    {'phone': 'a', 'start': 1.2, 'duration': 0.1, 'class': 'V'},
                                    {'phone': 'n', 'start': 1.3, 'duration': 0.1, 'class': 'C'},
                                    {'phone': 'c', 'start': 1.4, 'duration': 0.1, 'class': 'C'},
                                    {'phone': 'e', 'start': 1.5, 'duration': 0.1, 'class': 'V'}
                                ]
                            }
                        ]
                    }
                ]
            }
        }

    def test_extract_phone_data(self):
        """Test extraction of phone-level data."""
        # Extract phone data
        phone_df = extract_phone_data([self.test_utterance])
        
        # Check that all phones are extracted
        self.assertEqual(len(phone_df), 13)
        
        # Check that DataFrame has the expected columns
        expected_columns = [
            'utterance_id', 'utterance_text', 'word', 'word_idx', 
            'word_pos_in_utterance', 'word_start', 'word_duration', 
            'phone', 'phone_idx', 'phone_pos_in_word', 'phone_start', 
            'phone_duration', 'phone_class'
        ]
        for col in expected_columns:
            self.assertIn(col, phone_df.columns)
        
        # Check a few values
        self.assertEqual(phone_df['word'].iloc[0], 'TEST')
        self.assertEqual(phone_df['phone'].iloc[0], 't')
        self.assertEqual(phone_df['phone_duration'].iloc[0], 0.1)
        self.assertEqual(phone_df['phone_class'].iloc[0], 'C')

    def test_calculate_speaking_rate(self):
        """Test calculation of speaking rate features."""
        # Create a test DataFrame
        phone_df = extract_phone_data([self.test_utterance])
        
        # Calculate speaking rate
        result_df = calculate_speaking_rate(phone_df)
        
        # Check that speaking rate columns are added
        expected_columns = [
            'phone_duration_mean', 'phone_duration_std', 'speaking_rate',
            'phone_duration_norm'
        ]
        for col in expected_columns:
            self.assertIn(col, result_df.columns)
        
        # Check that values are calculated correctly
        self.assertAlmostEqual(result_df['phone_duration_mean'].iloc[0], 
                               phone_df['phone_duration'].mean(), places=4)
        
        # Speaking rate should be phones per second
        expected_rate = len(phone_df) / phone_df['phone_duration'].sum()
        self.assertAlmostEqual(result_df['speaking_rate'].iloc[0], expected_rate, places=4)


if __name__ == '__main__':
    unittest.main()