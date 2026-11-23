# Voice-Based Parkinson's Disease Detector

A machine-learning pipeline that screens for early signs of Parkinson's disease from a short voice recording. Built at **NeuroHack 2025** (Nucleate UK & ARIA), November 2025.

Parkinson's affects the muscles used for speech early on, often before the tremor is obvious. Small irregularities in a sustained vowel, such as cycle-to-cycle variation in pitch (jitter) and loudness (shimmer) and a lower harmonics-to-noise ratio, can be measured from a phone recording.

## How it works

```
Sustained "ahh" recording → noise clean-up → 18 voice features (Praat/Parselmouth) → SVM → healthy / Parkinson's + confidence
```

1. **Clean the audio** (`src/audio_cleaner*.py`): gentle, balanced and aggressive noise-reduction settings for recordings made in a noisy hackathon room.
2. **Extract features** (`src/audio_feature_extractor.py`): Praat via Parselmouth measures fundamental frequency, jitter, shimmer and harmonics-to-noise ratio, 18 features in total.
3. **Match the training format** (`src/extract_features.py`): maps the 18 features onto the UCI dataset's 22 columns and fills the six nonlinear measures it can't compute (RPDE, DFA, spread1, spread2, D2, PPE) with healthy-group statistics.
4. **Train** (`src/pd_detector.py`, `src/model_trainer.py`): standardised features, SVM with an RBF kernel, hyperparameters (C, gamma) tuned by `RandomizedSearchCV` with 5-fold cross-validation over 50 draws.
5. **Evaluate and predict** (`src/model_evaluator.py`, `check_person.py`).

## Data

| Source | Recordings | Used for |
|---|---|---|
| UCI Parkinson's dataset (Little et al., 2009) | 195 recordings from 31 people (23 with Parkinson's) | Training and testing the main model |
| Hackathon volunteers | 13 people, all healthy | Testing on real phone recordings |
| Additional labelled sustained-vowel recordings (`data/audio_10/`) | 14 (12 healthy, 2 Parkinson's) | Retraining a small custom model |

The volunteers' recordings are not published. In the CSV files they appear only as P01–P13.

## Results

| Test | Result |
|---|---|
| UCI held-out test set (20%, 39 recordings) | **92.3%** (36/39): all 32 Parkinson's recordings correct, 4 of 7 healthy recordings correct |
| Custom model on the 14 extra recordings, leave-one-out cross-validation | **92.9%** (13/14) |
| Hackathon volunteers (13 healthy) | All 13 classified healthy |

## Limitations

- **The UCI split is by recording, not by person.** Each person contributes about six recordings, so the same voice can appear in both the training and test sets, which makes the test accuracy optimistic. A split by person would be the honest next step.
- The healthy class is small (7 recordings in the test set), and 3 of those 7 were misclassified.
- Only 2 Parkinson's recordings outside the UCI data, and no volunteers with Parkinson's.
- Six UCI features are imputed for new recordings rather than measured.
- A screening prototype, not a diagnostic tool.

## Run it

```bash
pip install praat-parselmouth scikit-learn pandas numpy matplotlib joblib

python3 src/pd_detector.py        # prepare and split the UCI data
python3 src/model_trainer.py      # train the SVM
python3 src/model_evaluator.py    # accuracy, confusion matrix, F1

python3 check_person.py           # predict for a new recording
```

## Reference

Little, M.A., McSharry, P.E., Hunter, E.J., Spielman, J., Ramig, L.O. (2009). Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. *IEEE Transactions on Biomedical Engineering*, 56(4), 1015–1022. Dataset: UCI Machine Learning Repository.

## Tech stack

Python · Praat/Parselmouth · scikit-learn · pandas · NumPy · matplotlib
