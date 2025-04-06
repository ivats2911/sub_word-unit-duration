#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for subword unit duration modeling.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def plot_duration_histograms(df: pd.DataFrame, by_class: bool = False, save_path: Optional[str] = None):
    """
    Plot histograms of phone durations with consistent axes.
    
    Args:
        df: DataFrame with phone data
        by_class: Whether to separate by phone class
        save_path: If provided, saves the plot to this path
    """
    plt.figure(figsize=(12, 6))

    # Clip extreme durations to 99.5 percentile
    max_duration = df['phone_duration'].quantile(0.995)
    clipped_df = df[df['phone_duration'] <= max_duration]

    # Use dynamic bin range and bin width
    bin_min = 0
    bin_max = np.ceil(max_duration * 20) / 20  # round up to nearest 0.05
    bins = np.linspace(bin_min, bin_max, 50)  # 50 bins between min and max

    if by_class and 'phone_class' in clipped_df.columns:
        classes = clipped_df['phone_class'].unique()
        for cls in classes:
            group = clipped_df[clipped_df['phone_class'] == cls]
            sns.histplot(
                group['phone_duration'],
                label=cls,
                bins=bins,
                kde=True,
                element='step',
                alpha=0.5,
                common_norm=False
            )
        plt.legend()
        plt.title('Phone Duration Distribution by Class')
    else:
        sns.histplot(
            clipped_df['phone_duration'],
            bins=bins,
            kde=True,
            color='steelblue',
            element='step'
        )
        plt.title('Overall Phone Duration Distribution')

    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()




def plot_top_phones_by_duration(df: pd.DataFrame, n: int = 20, save_path: Optional[str] = None):
    """
    Plot average durations for top N most frequent phones.
    
    Args:
        df: DataFrame with phone data
        n: Number of top phones to plot
        save_path: Path to save the plot
    """
    # Get counts and mean durations for each phone
    phone_stats = df.groupby('phone').agg({
        'phone_duration': ['count', 'mean', 'std']
    })
    
    phone_stats.columns = ['_'.join(col).strip() for col in phone_stats.columns.values]
    phone_stats = phone_stats.reset_index()
    
    # Sort by count and get top N
    top_phones = phone_stats.sort_values('phone_duration_count', ascending=False).head(n)
    
    # Sort by duration for the plot
    top_phones = top_phones.sort_values('phone_duration_mean')
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bars with error bars
    plt.barh(
        top_phones['phone'],
        top_phones['phone_duration_mean'],
        xerr=top_phones['phone_duration_std'],
        alpha=0.7,
        capsize=5
    )
    
    # Add count annotations
    for i, (_, row) in enumerate(top_phones.iterrows()):
        plt.text(
            row['phone_duration_mean'] + row['phone_duration_std'] + 0.01,
            i,
            f"n={int(row['phone_duration_count'])}",
            va='center'
        )
    
    plt.xlabel('Mean Duration (s)')
    plt.ylabel('Phone')
    plt.title(f'Average Duration of Top {n} Phones by Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_position_effects(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot effects of position on phone duration.
    
    Args:
        df: DataFrame with phone data
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Effect of position in word
    ax1.scatter(df['phone_pos_in_word'], df['phone_duration'], alpha=0.1)
    
    # Add trend line
    bins = np.linspace(0, 1, 11)
    bin_centers = bins[:-1] + np.diff(bins)/2
    bin_means = [df[(df['phone_pos_in_word'] >= bins[i]) & 
                   (df['phone_pos_in_word'] < bins[i+1])]['phone_duration'].mean() 
                for i in range(len(bins)-1)]
    
    ax1.plot(bin_centers, bin_means, 'r-', linewidth=2)
    
    ax1.set_xlabel('Position in Word (0=beginning, 1=end)')
    ax1.set_ylabel('Duration (s)')
    ax1.set_title('Effect of Position in Word on Phone Duration')
    ax1.grid(True, alpha=0.3)
    
    # 2. Effect of position in utterance
    ax2.scatter(df['word_pos_in_utterance'], df['phone_duration'], alpha=0.1)
    
    # Add trend line
    bins = np.linspace(0, 1, 11)
    bin_centers = bins[:-1] + np.diff(bins)/2
    bin_means = [df[(df['word_pos_in_utterance'] >= bins[i]) & 
                   (df['word_pos_in_utterance'] < bins[i+1])]['phone_duration'].mean() 
                for i in range(len(bins)-1)]
    
    ax2.plot(bin_centers, bin_means, 'r-', linewidth=2)
    
    ax2.set_xlabel('Position in Utterance (0=beginning, 1=end)')
    ax2.set_ylabel('Duration (s)')
    ax2.set_title('Effect of Position in Utterance on Phone Duration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_speaking_rate_effects(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot effects of speaking rate on phone duration.
    
    Args:
        df: DataFrame with phone data
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with trend line
    sns.regplot(x='speaking_rate', y='phone_duration', data=df, scatter_kws={'alpha': 0.1})
    
    plt.xlabel('Speaking Rate (phones/second)')
    plt.ylabel('Phone Duration (s)')
    plt.title('Effect of Speaking Rate on Phone Duration')
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlation
    corr = df[['speaking_rate', 'phone_duration']].corr().iloc[0, 1]
    plt.annotate(f'Correlation: {corr:.3f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_context_effects(df: pd.DataFrame, phone: str, context_size: int = 1, save_path: Optional[str] = None):
    """
    Plot effects of surrounding phones on duration of a specific phone.
    
    Args:
        df: DataFrame with phone data
        phone: Phone to analyze
        context_size: Size of context window
        save_path: Path to save the plot
    """
    # Filter data for the specified phone
    phone_df = df[df['phone'] == phone].copy()
    
    if len(phone_df) < 10:
        logger.warning(f"Not enough data for phone '{phone}' (found {len(phone_df)} samples)")
        return
    
    # Ensure context columns exist
    context_cols = []
    for i in range(1, context_size + 1):
        prev_col = f'prev_{i}_phone'
        next_col = f'next_{i}_phone'
        
        if prev_col not in phone_df.columns or next_col not in phone_df.columns:
            logger.warning(f"Context columns {prev_col} or {next_col} not found in DataFrame")
            return
        
        context_cols.extend([prev_col, next_col])
    
    # Create figure with subplots for each context position
    fig, axes = plt.subplots(len(context_cols), 1, figsize=(12, 4 * len(context_cols)))
    
    if len(context_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(context_cols):
        # Get average duration by context phone
        context_stats = phone_df.groupby(col)['phone_duration'].agg(['mean', 'std', 'count'])
        
        # Filter to include only context phones with sufficient samples
        context_stats = context_stats[context_stats['count'] >= 5]
        
        # Sort by mean duration
        context_stats = context_stats.sort_values('mean')
        
        # Plot
        axes[i].barh(
            context_stats.index,
            context_stats['mean'],
            xerr=context_stats['std'],
            alpha=0.7,
            capsize=5
        )
        
        # Add count annotations
        for j, (ctx_phone, row) in enumerate(context_stats.iterrows()):
            axes[i].text(
                row['mean'] + row['std'] + 0.01,
                j,
                f"n={int(row['count'])}",
                va='center'
            )
        
        axes[i].set_xlabel('Mean Duration (s)')
        axes[i].set_ylabel(col)
        axes[i].set_title(f'Effect of {col} on Duration of "{phone}"')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_native_vs_nonnative(native_df: pd.DataFrame, nonnative_df: pd.DataFrame, 
                            feature: str = 'phone_duration', by: str = 'phone_class',
                            save_path: Optional[str] = None):
    """
    Compare feature distributions between native and non-native speakers.
    
    Args:
        native_df: DataFrame with native speaker data
        nonnative_df: DataFrame with non-native speaker data
        feature: Feature to compare
        by: Variable to group by
        save_path: Path to save the plot
    """
    # Add speaker type column
    native_df = native_df.copy()
    native_df['speaker_type'] = 'Native'
    
    nonnative_df = nonnative_df.copy()
    nonnative_df['speaker_type'] = 'Non-native'
    
    # Combine data
    combined_df = pd.concat([native_df, nonnative_df])
    
    plt.figure(figsize=(12, 6))
    
    # Create box plot
    sns.boxplot(x=by, y=feature, hue='speaker_type', data=combined_df)
    
    plt.xlabel(by.capitalize())
    plt.ylabel(feature.capitalize())
    plt.title(f'Comparison of {feature} between Native and Non-native Speakers by {by}')
    plt.xticks(rotation=45)
    plt.legend(title='Speaker Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(model, feature_names: List[str], top_n: int = 20, save_path: Optional[str] = None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to plot
        save_path: Path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance and get top N
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    sns.barplot(x='importance', y='feature', data=importance_df)
    
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()