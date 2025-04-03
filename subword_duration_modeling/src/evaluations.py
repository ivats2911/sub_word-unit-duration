#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics and functions for subword unit duration modeling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True durations
        y_pred: Predicted durations
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True durations
        y_pred: Predicted durations
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def correlation_coefficient(y_true, y_pred):
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_true: True durations
        y_pred: Predicted durations
        
    Returns:
        Correlation coefficient
    """
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def evaluate_model(model, X, y, config=None, phone_mapping=None):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: Target durations
        config: Configuration object
        phone_mapping: Mapping from indices to phone labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    metrics = {}
    metrics['mae'] = mean_absolute_error(y, predictions)
    metrics['rmse'] = root_mean_squared_error(y, predictions)
    metrics['correlation'] = correlation_coefficient(y, predictions)
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"Correlation: {metrics['correlation']:.4f}")
    
    return metrics


def evaluate_by_phone_class(model, X, y, phone_class, config=None):
    """
    Evaluate model performance by phone class.
    
    Args:
        model: Trained model
        X: Features
        y: Target durations
        phone_class: Array with phone class labels ('C', 'V', 'sil')
        config: Configuration object
        
    Returns:
        Dictionary with evaluation metrics by phone class
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics by phone class
    metrics_by_class = {}
    
    for cls in np.unique(phone_class):
        mask = phone_class == cls
        y_cls = y[mask]
        pred_cls = predictions[mask]
        
        metrics_cls = {}
        metrics_cls['mae'] = mean_absolute_error(y_cls, pred_cls)
        metrics_cls['rmse'] = root_mean_squared_error(y_cls, pred_cls)
        metrics_cls['correlation'] = correlation_coefficient(y_cls, pred_cls)
        metrics_cls['count'] = np.sum(mask)
        
        metrics_by_class[cls] = metrics_cls
        
        logger.info(f"Class {cls} ({metrics_cls['count']} samples):")
        logger.info(f"  MAE: {metrics_cls['mae']:.4f}")
        logger.info(f"  RMSE: {metrics_cls['rmse']:.4f}")
        logger.info(f"  Correlation: {metrics_cls['correlation']:.4f}")
    
    return metrics_by_class


def plot_predictions(y_true, y_pred, title='Predicted vs Actual Durations', save_path=None):
    """
    Plot predicted durations against actual durations.
    
    Args:
        y_true: True durations
        y_pred: Predicted durations
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Duration (s)')
    plt.ylabel('Predicted Duration (s)')
    plt.title(title)
    
    # Add metrics to plot
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    corr = correlation_coefficient(y_true, y_pred)
    
    plt.annotate(f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nCorr: {corr:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_error_distribution(y_true, y_pred, title='Error Distribution', save_path=None):
    """
    Plot distribution of prediction errors.
    
    Args:
        y_true: True durations
        y_pred: Predicted durations
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    
    plt.xlabel('Prediction Error (s)')
    plt.ylabel('Frequency')
    plt.title(title)
    
    # Add mean and std to plot
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    plt.axvline(mean_error, color='r', linestyle='--')
    plt.annotate(f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_evaluation_report(model, X, y, phone_ids=None, phone_classes=None, config=None, save_dir=None):
    """
    Create a comprehensive evaluation report.
    
    Args:
        model: Trained model
        X: Features
        y: Target durations
        phone_ids: Array with phone IDs
        phone_classes: Array with phone class labels
        config: Configuration object
        save_dir: Directory to save plots and reports
        
    Returns:
        Dictionary with evaluation results
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Overall metrics
    overall_metrics = {
        'mae': mean_absolute_error(y, predictions),
        'rmse': root_mean_squared_error(y, predictions),
        'correlation': correlation_coefficient(y, predictions),
        'count': len(y)
    }
    
    # Metrics by phone class (if available)
    class_metrics = None
    if phone_classes is not None:
        class_metrics = evaluate_by_phone_class(model, X, y, phone_classes, config)
    
    # Metrics by phone (if available)
    phone_metrics = None
    if phone_ids is not None:
        unique_phones = np.unique(phone_ids)
        phone_metrics = {}
        
        for phone in unique_phones:
            mask = phone_ids == phone
            y_phone = y[mask]
            pred_phone = predictions[mask]
            
            if len(y_phone) >= 5:  # Only evaluate phones with sufficient samples
                phone_metrics[phone] = {
                    'mae': mean_absolute_error(y_phone, pred_phone),
                    'rmse': root_mean_squared_error(y_phone, pred_phone),
                    'correlation': correlation_coefficient(y_phone, pred_phone) if len(y_phone) > 5 else np.nan,
                    'count': len(y_phone)
                }
    
    # Generate plots
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Predicted vs Actual plot
        plot_predictions(y, predictions, 
                        title='Predicted vs Actual Durations',
                        save_path=os.path.join(save_dir, 'predictions.png'))
        
        # Error distribution plot
        plot_error_distribution(y, predictions,
                               title='Error Distribution',
                               save_path=os.path.join(save_dir, 'error_distribution.png'))
        
        # Save metrics as CSV
        results_df = pd.DataFrame([overall_metrics])
        results_df.to_csv(os.path.join(save_dir, 'overall_metrics.csv'), index=False)
        
        if class_metrics:
            class_df = pd.DataFrame.from_dict(class_metrics, orient='index')
            class_df.to_csv(os.path.join(save_dir, 'class_metrics.csv'))
        
        if phone_metrics:
            phone_df = pd.DataFrame.from_dict(phone_metrics, orient='index')
            phone_df.to_csv(os.path.join(save_dir, 'phone_metrics.csv'))
    
    # Return all metrics
    return {
        'overall': overall_metrics,
        'by_class': class_metrics,
        'by_phone': phone_metrics
    }