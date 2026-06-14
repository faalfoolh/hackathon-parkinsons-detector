"""
Retrain model on custom labeled data
Small dataset - use careful cross-validation
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("RETRAINING MODEL ON YOUR LABELED DATA")
print("=" * 70)

# Load your labeled data
df = pd.read_csv('data/voice_features_database.csv')

# Separate features and labels
X = df.drop(columns=['name', 'status', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'])
y = df['status'].fillna(0).astype(int)

print(f"\nDataset: {len(df)} samples")
print(f"  Healthy: {(y == 0).sum()}")
print(f"  Parkinson's: {(y == 1).sum()}")

# Handle any missing values
X = X.fillna(X.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 70)
print("TRAINING WITH LEAVE-ONE-OUT CROSS-VALIDATION")
print("(Small dataset - each sample tested once)")
print("=" * 70)

# Train SVM with Leave-One-Out CV
# Using simpler parameters due to small dataset
svm = SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42)

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
cv_scores = cross_val_score(svm, X_scaled, y, cv=loo, scoring='accuracy')

print(f"\nLeave-One-Out CV Accuracy: {cv_scores.mean():.2%}")
print(f"Correctly classified: {int(cv_scores.sum())}/{len(y)}")

# Train final model on all data
svm.fit(X_scaled, y)

# Make predictions
y_pred = svm.predict(X_scaled)
y_proba = svm.predict_proba(X_scaled)

print("\n" + "=" * 70)
print("RESULTS ON YOUR DATA:")
print("=" * 70)

# Show confusion matrix
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(f"                 Predicted Healthy  Predicted PD")
print(f"Actually Healthy      {cm[0,0]:5d}           {cm[0,1]:5d}")
print(f"Actually PD           {cm[1,0]:5d}           {cm[1,1]:5d}")

# Classification report
print("\n" + classification_report(y, y_pred, target_names=['Healthy', 'Parkinson\'s']))

# Individual predictions
results_df = pd.DataFrame({
    'Name': df['name'],
    'Actual': y,
    'Predicted': y_pred,
    'Confidence_Healthy': [p[0] * 100 for p in y_proba],
    'Confidence_PD': [p[1] * 100 for p in y_proba]
})

results_df['Status'] = results_df.apply(
    lambda row: '✓ Correct' if row['Actual'] == row['Predicted'] else '✗ WRONG',
    axis=1
)

print("\n" + "=" * 70)
print("INDIVIDUAL PREDICTIONS:")
print("=" * 70)
for _, row in results_df.iterrows():
    actual = "PD" if row['Actual'] == 1 else "Healthy"
    predicted = "PD" if row['Predicted'] == 1 else "Healthy"
    status = row['Status']
    conf_pd = row['Confidence_PD']

    print(f"{row['Name'][:20]:<20} Actual: {actual:<8} Predicted: {predicted:<8} ({conf_pd:.1f}% PD) {status}")

# Save retrained model and scaler
joblib.dump(svm, 'model/custom_svm_model.pkl')
joblib.dump(scaler, 'model/custom_scaler.pkl')

print("\n" + "=" * 70)
print("✓ Custom model saved to: model/custom_svm_model.pkl")
print("✓ Custom scaler saved to: model/custom_scaler.pkl")
print("=" * 70)
