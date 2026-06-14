# Hackathon — Voice-Based Parkinson's Disease Detector

A machine learning system that detects early signs of Parkinson's Disease from voice recordings, built at a hackathon in November 2025.

## How it works
1. Record a short voice sample (sustained "ahh" sound)
2. Extract 18 acoustic features (jitter, shimmer, HNR, etc.)
3. SVM classifier predicts: Healthy or Parkinson's

## Pipeline
```
Voice Recording → Feature Extraction → SVM Classifier → Prediction
```

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `src/pd_detector.py` | Load & scale UCI dataset, 80/20 split |
| 2 | `src/model_trainer.py` | Train SVM (RBF kernel, RandomizedSearchCV) |
| 3 | `src/model_evaluator.py` | Accuracy, confusion matrix, F1 score |
| 4 | `src/feature_plotter.py` | Visualise healthy vs PD features |

## Dataset
- UCI Parkinson's Dataset (195 samples, 22 voice features)
- Custom recordings from 13 participants

## Tech Stack
- Python 3.11
- scikit-learn (SVM, StandardScaler, RandomizedSearchCV)
- librosa (audio feature extraction)
- pandas, numpy, matplotlib

## Results
SVM with RBF kernel, 5-fold cross-validation, 50 hyperparameter iterations.

## Run it
```bash
# Train the model
python3 src/pd_detector.py
python3 src/model_trainer.py

# Evaluate
python3 src/model_evaluator.py

# Predict on new voice
python3 check_person.py
```
