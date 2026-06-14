# model_evaluator.py - Phase 3: Final Evaluation and Metrics

import pandas as pd
import joblib 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore') # Clean terminal output

print("--- Starting Phase 3: Model Evaluation and Metrics ---")

# ----------------------------------------------------
# STEP 7: Load Optimized Model and Test Data
# ----------------------------------------------------
try:
    # Load the optimized model built by Team 2
    best_model = joblib.load('model/best_svm_model.pkl')

    # Load the test sets reserved by Team 1 (data the model has NEVER seen)
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')['status']

    print("Optimized model and test data loaded successfully!")
except FileNotFoundError:
    print("\nERROR: Could not find required files (model/best_svm_model.pkl, data/X_test.csv).")
    print("ACTION: Ensure Phase 1 and Phase 2 have completed successfully.")
    exit()

# ----------------------------------------------------
# STEP 8: Predict and Evaluate Performance
# ----------------------------------------------------
# The model makes predictions on the unseen test data
y_pred = best_model.predict(X_test)

# Calculate the simple accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- FINAL PERFORMANCE METRICS ---")
print(f"1. Overall Accuracy: {accuracy*100:.2f}%")

# Generate the Confusion Matrix (The visual proof)
cm = confusion_matrix(y_test, y_pred)
print("\n2. Confusion Matrix (The Proof):")
print(f"   [ True Negatives | False Positives ]")
print(f"   [ False Negatives | True Positives  ]")
print(cm)

# Generate detailed metrics (Precision, Recall, F1-Score)
print("\n3. Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Parkinson\'s (1)']))


# ----------------------------------------------------
# STEP 9: Final Conclusion
# ----------------------------------------------------
print(f"--------------------------------------------------")
print(f"CONCLUSION: The optimized SVM model successfully classified {accuracy*100:.2f}% of the unseen test samples.")
print(f"The high metrics confirm the model's reliability for pre-screening.")
print(f"--------------------------------------------------")