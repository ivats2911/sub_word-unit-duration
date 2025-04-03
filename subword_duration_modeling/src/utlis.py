#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for subword unit duration modeling.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_logging(log_dir='logs', level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'app.log')),
            logging.StreamHandler()
        ]
    )


def save_json(data, filepath):
    """
    Save data as JSON.
    
    Args:
        data: Data to save
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load data from JSON.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_duration_distribution(df, column='phone_duration', by=None, n_top=None, save_path=None):
    """
    Plot distribution of durations.
    
    Args:
        df: DataFrame with duration data
        column: Column name for durations
        by: Column to group by (optional)
        n_top: Number of top categories to plot (if grouped)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    if by is None:
        # Overall distribution
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
    else:
        # Group by a category
        if n_top is not None:
            # Get top N categories by frequency
            top_cats = df[by].value_counts().nlargest(n_top).index
            filtered_df = df[df[by].isin(top_cats)]
        else:
            filtered_df = df
        
        # Plot distribution by category
        sns.boxplot(x=by, y=column, data=filtered_df)
        plt.title(f'Distribution of {column} by {by}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(df, columns=None, save_path=None):
    """
    Plot correlation matrix for selected columns.
    
    Args:
        df: DataFrame with data
        columns: List of columns to include (optional)
        save_path: Path to save the plot (optional)
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    corr = df[columns].corr()
    
    # Plot
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_speaking_rate_metrics(df, group_by='utterance_id'):
    """
    Calculate speaking rate metrics.
    
    Args:
        df: DataFrame with phone data
        group_by: Column to group by
        
    Returns:
        DataFrame with speaking rate metrics
    """
    # Group by the specified column
    grouped = df.groupby(group_by)
    
    # Calculate metrics
    metrics = grouped.agg({
        'phone_duration': ['count', 'sum', 'mean', 'std'],
        'phone': 'count'
    })
    
    # Clean up column names
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
    
    # Calculate speaking rate (phones per second)
    metrics['speaking_rate'] = metrics['phone_count'] / metrics['phone_duration_sum']
    
    # Calculate coefficient of variation (normalized std)
    metrics['duration_cv'] = metrics['phone_duration_std'] / metrics['phone_duration_mean']
    
    return metrics.reset_index()


def compare_native_nonnative(native_df, nonnative_df, column='phone_duration', by='phone', 
                            top_n=20, save_path=None):
    """
    Compare durations between native and non-native speakers.
    
    Args:
        native_df: DataFrame with native speaker data
        nonnative_df: DataFrame with non-native speaker data
        column: Column to compare
        by: Column to group by
        top_n: Number of top categories to compare
        save_path: Path to save the plot (optional)
    """
    # Get top N categories by frequency in native data
    top_cats = native_df[by].value_counts().nlargest(top_n).index
    
    # Filter data to include only top categories
    native_filtered = native_df[native_df[by].isin(top_cats)]
    nonnative_filtered = nonnative_df[nonnative_df[by].isin(top_cats)]
    
    # Calculate statistics for each category
    native_stats = native_filtered.groupby(by)[column].agg(['mean', 'std']).reset_index()
    native_stats.columns = [by, 'native_mean', 'native_std']
    
    nonnative_stats = nonnative_filtered.groupby(by)[column].agg(['mean', 'std']).reset_index()
    nonnative_stats.columns = [by, 'nonnative_mean', 'nonnative_std']
    
    # Merge statistics
    stats = pd.merge(native_stats, nonnative_stats, on=by)
    
    # Calculate difference
    stats['diff'] = stats['nonnative_mean'] - stats['native_mean']
    stats['diff_pct'] = (stats['diff'] / stats['native_mean']) * 100
    
    # Sort by absolute difference
    stats = stats.sort_values('diff_pct', key=abs, ascending=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(stats))
    width = 0.35
    
    plt.bar(x - width/2, stats['native_mean'], width, label='Native', alpha=0.7)
    plt.bar(x + width/2, stats['nonnative_mean'], width, label='Non-native', alpha=0.7)
    
    plt.xticks(x, stats[by], rotation=45, ha='right')
    plt.xlabel(by.capitalize())
    plt.ylabel(f'Mean {column}')
    plt.title(f'Comparison of {column} between Native and Non-native Speakers')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage difference on top of bars
    for i, (_, row) in enumerate(stats.iterrows()):
        plt.text(i, max(row['native_mean'], row['nonnative_mean']) + 0.01, 
                f"{row['diff_pct']:.1f}%", 
                ha='center', va='bottom', 
                color='green' if row['diff_pct'] > 0 else 'red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return stats