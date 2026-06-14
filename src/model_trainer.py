# model_trainer.py - Phase 2: SVM Optimization

import pandas as pd
import joblib 
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, reciprocal
import warnings
warnings.filterwarnings('ignore') # Suppresses minor warnings for a clean hackathon terminal output

print("--- Starting Phase 2: Model Training and Optimization ---")

# ----------------------------------------------------
# STEP 4: Load Data from Phase 1 Handoff
# ----------------------------------------------------
try:
    # Load the training sets created by Team 1
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')['status'] # Load only the 'status' column as the target

    print("Training data loaded successfully!")
except FileNotFoundError:
    print("\nERROR: Could not find training data files (data/X_train.csv, data/y_train.csv).")
    print("ACTION: Team 1 must run pd_detector.py first and share the resulting CSVs!")
    exit()

# ----------------------------------------------------
# STEP 5: Hyperparameter Tuning (Optimization)
# ----------------------------------------------------
# Initialize the SVM model
svm = SVC(kernel='rbf', random_state=42, probability=True)

# Define the search space for the best C (regularization) and gamma (kernel width)
# We are searching for the best control settings for the SVM.
param_distributions = {
    'C': uniform(1, 100),         # Search C values between 1 and 100
    'gamma': reciprocal(0.001, 1) # Search gamma values between 0.001 and 1
}

# Set up the Randomized Search
# n_iter=50 tries 50 different combinations. n_jobs=-1 uses all CPU cores for speed!
random_search = RandomizedSearchCV(
    svm,
    param_distributions=param_distributions,
    n_iter=50,                  
    cv=5,                       
    random_state=42,
    n_jobs=-1,                  
    scoring='accuracy'
)

print("\nStarting Hyperparameter Optimization (this will take a few minutes)...")
random_search.fit(X_train, y_train)

# Get the best model found
best_model = random_search.best_estimator_

print("\nOptimization Complete!")
print(f"Best SVM Parameters Found: {random_search.best_params_}")
print(f"Accuracy of Best Model on Training Data (Cross-Validation Score): {random_search.best_score_:.4f}")

# ----------------------------------------------------
# STEP 6: Save the Best Model (Handoff to Team 3)
# ----------------------------------------------------
# joblib saves the trained model so Team 3 doesn't have to re-train it.
joblib.dump(best_model, 'model/best_svm_model.pkl')

# Also save information about the training data for proper scaling later
print("\nSaving model metadata for predictions...")
model_metadata = {
    'feature_names': X_train.columns.tolist(),
    'n_features': len(X_train.columns)
}
joblib.dump(model_metadata, 'model/model_metadata.pkl')

print("\n--- Phase 2 COMPLETE: Saved optimized model to 'model/best_svm_model.pkl' ---")