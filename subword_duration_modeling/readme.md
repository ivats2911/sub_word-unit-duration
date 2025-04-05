# Subword Unit Duration Modeling

This project focuses on modeling the duration of subword units (such as phonemes) in speech. It uses time alignment data from an Automatic Speech Recognition (ASR) system to predict the duration of each subword unit in a given sentence. This can be used to assess how closely an L2 speaker's speech characteristics resemble those of native speakers.

## Project Overview

The goal is to develop a model that can:
1. Extract subword unit durations from ASR-provided time alignment files
2. Model the duration of each subword unit for prediction
3. Analyze differences between native and non-native speakers

## Installation

### Requirements
- Python 3.8+
- Required libraries are listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/ivats2911/sub_word-unit-duration.git
cd subword_duration_modeling

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data
# Data should be placed in the data directory with the following structure:
# data/
#   american_english/  # Native speaker data
#   other_english/     # Non-native speaker data
```

## Project Structure

```
subword_duration_modeling/
├── data/                      # Data directory
│   ├── american_english/      # Native speaker data
│   └── other_english/         # Non-native speaker data
├── src/                       # Source code
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── features.py            # Feature engineering
│   ├── models.py              # Duration models (Baseline, Linear, RF, XGBoost, LSTM)
│   ├── evaluation.py          # Evaluation metrics
│   ├── utils.py               # Utility functions
│   └── visualization.py       # Visualization utilities
├── notebooks/                 # Jupyter notebooks for exploration
├── models/                    # Saved models
├── reports/                   # Reports and figures
│   └── figures/               # Generated figures (auto-saved)
├── tests/                     # Unit tests for model logic
├── main.py                    # Main script to run the pipeline
├── config.py                  # Configuration settings
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Usage

### Data Preparation

The data consists of JSON files with ASR time alignment results. The `data_processing.py` module extracts word and phone-level segments, filters OOV tokens, and generates aligned features.

### Model Training

The project implements several models for duration prediction:
- Baseline model using phone-wise mean duration
- Linear regression
- Random Forest (best performing)
- XGBoost
- LSTM (planned, but not finalized due to sparse input issues)

Example to train a model:
```bash
python main.py --data_dir data --output_dir models --model_type rf --train
```

Available model types: `baseline`, `linear`, `rf`, `xgboost`, `lstm`

### Evaluation

```bash
python main.py --data_dir data --output_dir models --model_type rf --evaluate
```

Evaluation metrics and plots are saved to the `reports/figures` directory automatically.

## Features

The features extracted include:
- Phone identity (one-hot encoded)
- Phone class (consonant, vowel, silence)
- Context phones (previous and next N phones)
- Context classes
- Normalized word and sentence position
- Speaking rate (phones/sec)

## Model Details

### Baseline Model
Uses average phone duration from training data. Laplace smoothing is applied.

### Linear Model
Linear regression that maps input features to phone durations.

### Tree-based Models
Random Forest and XGBoost capture non-linearities in contextual and positional features.

### LSTM Model
LSTM model structure was implemented but training failed due to sparse input matrix format. This is marked for future work.

## Evaluation Metrics

Models are evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Correlation (Pearson)

## Native vs Non-native Analysis

The best model (Random Forest) showed:
- MAE (native speakers) < MAE (non-native speakers)
- Distinct duration patterns across speaker groups
- Feature importance revealed: phone identity, speaking rate, and word position are highly influential

## Future Work

- Fix LSTM to support sparse-to-dense conversion
- Integrate prosodic/syllabic features
- Explore transformer-based sequence models
- Build feedback tools for pronunciation training

## Acknowledgements

This project uses a subset of the VoxForge dataset and builds on research in ASR alignment, phonetic modeling, and second language speech.

## License

[MIT License](LICENSE)