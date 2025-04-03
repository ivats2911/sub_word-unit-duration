#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for the subword unit duration modeling project.
This script orchestrates the entire pipeline from data loading to model evaluation.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data_processing import load_data, preprocess_data
from src.features import extract_features
from src.models import train_model, load_model
from src.evaluation import evaluate_model
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Subword Unit Duration Modeling')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the datasets')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models and results')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'linear', 'rf', 'xgboost', 'lstm', 'transformer'],
                        help='Type of model to train')
    parser.add_argument('--train', action='store_true',
                        help='Train a new model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the model on test data')
    parser.add_argument('--native_only', action='store_true',
                        help='Use only native speaker data for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Parse arguments
    args = parse_args()
    logger.info(f"Starting pipeline with arguments: {args}")
    
    # Initialize configuration
    config = Config()
    config.update_from_args(args)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    native_data = load_data(os.path.join(args.data_dir, 'american_english'))
    non_native_data = load_data(os.path.join(args.data_dir, 'other_english'))
    
    train_data, val_data, test_data = preprocess_data(
        native_data, 
        non_native_data,
        native_only=args.native_only,
        config=config
    )
    
    # Extract features
    logger.info("Extracting features...")
    X_train, y_train = extract_features(train_data, config)
    X_val, y_val = extract_features(val_data, config)
    X_test, y_test = extract_features(test_data, config)
    
    # Train or load model
    if args.train:
        logger.info(f"Training {args.model_type} model...")
        model = train_model(X_train, y_train, X_val, y_val, model_type=args.model_type, config=config)
        
        # Save the model
        model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
        logger.info(f"Saving model to {model_path}")
        model.save(model_path)
    else:
        model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
    
    # Evaluate model
    if args.evaluate:
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test, config)
        logger.info(f"Evaluation metrics: {metrics}")
    
    logger.info("Pipeline completed successfully")


if __name__ == '__main__':
    main()