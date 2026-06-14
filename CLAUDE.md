# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Parkinson's Disease (PD) detection system that uses Support Vector Machine (SVM) classification on voice biomarker features. The project analyzes vocal characteristics to classify patients as healthy or showing signs of Parkinson's Disease.

## Architecture

The codebase is structured as a **4-phase ML pipeline** with clear handoff points between phases:

### Phase 1: Data Preparation (`src/pd_detector.py`)
- Loads the UCI Parkinson's dataset from `data/parkinsons.csv` (or `parkinsons/parkinsons.csv`)
- Handles UCI-specific CSV formatting (header in first data row)
- Drops patient ID (`name`) column and extracts target (`status`: 0=healthy, 1=Parkinson's)
- Applies StandardScaler normalization to all 22 voice features
- Splits data 80/20 into train/test sets (random_state=42)
- **Outputs**: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` (saved to project root by default)

### Phase 2: Model Training (`src/model_trainer.py`)
- Loads Phase 1 CSV outputs
- Trains SVM with RBF kernel using RandomizedSearchCV hyperparameter optimization
- Search space: C ∈ [1, 100], gamma ∈ [0.001, 1]
- Uses 5-fold cross-validation with 50 iterations
- **Outputs**: `best_svm_model.pkl` (saved to `model/` directory)

### Phase 3: Model Evaluation (`src/model_evaluator.py`)
- Loads trained model and test data
- Generates performance metrics: accuracy, confusion matrix, classification report
- Reports precision, recall, F1-score for both classes

### Phase 4: Visualization (`src/feature_plotter.py`)
- Generates box plots comparing healthy vs PD patients
- Default features: `MDVP:Jitter(%)` and `MDVP:Shimmer`
- Uses matplotlib for visualization

### Feature Extraction (`src/extract_features.py`)
- Processes custom user voice recordings
- Takes 18-feature input data and maps to the 23-feature Parkinson's dataset format
- Uses **positional renaming** to map user columns to template columns
- Imputes 5 missing features (RPDE, DFA, spread1, spread2, D2, PPE) using statistics from healthy patients (status=0)
- **Outputs**:
  - `voice_features_database_GROUND_TRUTH.csv` (with status=0)
  - `voice_features_database_PREDICTION_INPUT.csv` (without status column)

## Key Data Files

### Input Data
- `data/parkinsons.csv`: UCI Parkinson's dataset (24 columns: name + 22 features + status)
- `data/voice_features_database.csv`: User-provided voice features (Name + 18 features + Label)
- `data/audio/`: Directory containing audio recordings (.m4a files)

### Generated Data
- `data/X_train.csv`, `data/X_test.csv`: Scaled feature sets
- `data/y_train.csv`, `data/y_test.csv`: Target labels
- `data/voice_features_database_GROUND_TRUTH.csv`: Aligned features with known status
- `data/voice_features_database_PREDICTION_INPUT.csv`: Features ready for model prediction

### Model Artifacts
- `model/best_svm_model.pkl`: Optimized SVM classifier (saved with joblib)

## Development Commands

### Running the Full Pipeline
Execute phases sequentially:
```bash
# Phase 1: Prepare data
python3 src/pd_detector.py

# Phase 2: Train and optimize model
python3 src/model_trainer.py

# Phase 3: Evaluate model
python3 src/model_evaluator.py

# Phase 4: Generate visualizations
python3 src/feature_plotter.py
```

### Processing User Voice Data
```bash
python3 src/extract_features.py
```

## Important Implementation Details

### Dataset Column Mapping
The UCI Parkinson's dataset has 24 columns:
1. `name` (patient ID, dropped during processing)
2-23. Voice features (MDVP metrics, jitter, shimmer, NHR, HNR, RPDE, DFA, spread, D2, PPE)
24. `status` (target: 0=healthy, 1=Parkinson's)

### UCI CSV Format Handling
The Parkinson's dataset has non-standard formatting where the header is embedded as the first data row. All scripts handle this with:
```python
df = pd.read_csv(FILE_PATH, header=None)
df.columns = df.iloc[0]
df = df[1:].copy()
```

### File Path Assumptions
- `pd_detector.py` expects data at `parkinsons/parkinsons.csv`
- `feature_plotter.py` expects data at `parkinsons/parkinsons.csv`
- Other scripts expect data in `data/` directory
- Model output goes to `model/` directory
- Train/test splits are saved to project root by default

### Reproducibility
- All random operations use `random_state=42`
- Train/test split is always 80/20
- Feature scaling uses StandardScaler fit on training data

## Dependencies

The project uses Python 3.11+ with:
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: ML models, preprocessing, evaluation
- scipy: Statistical distributions for hyperparameter search
- matplotlib: Visualization
- joblib: Model serialization

## Notes on Feature Engineering

The `extract_features.py` script bridges the gap between user-provided features (18 columns) and the model's expected format (23 columns):
- First 18 user features are positionally mapped to template columns
- Remaining 5 features are imputed using normal distribution based on healthy patient statistics
- This allows the model to make predictions on incomplete feature sets
