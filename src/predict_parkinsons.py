"""
Parkinson's Disease Prediction Script
Uses the trained SVM model to predict PD status for voice samples
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PARKINSON'S DISEASE PREDICTION")
print("=" * 70)

# Load the trained model
print("\n1. Loading trained SVM model...")
model = joblib.load('model/best_svm_model.pkl')

# Load the SCALER that was used during training (CRITICAL!)
print("2. Loading scaler from training...")
try:
    scaler = joblib.load('model/scaler.pkl')
    print("   ✓ Using original training scaler")
except FileNotFoundError:
    print("   ⚠️  WARNING: Original scaler not found!")
    print("   You need to run: python src/pd_detector.py first")
    exit(1)

# Load the prediction input data (without status column)
print("3. Loading voice features data...")
prediction_data = pd.read_csv('data/voice_features_database_PREDICTION_INPUT.csv')

# Extract names for later
names = prediction_data['name'].values

# Drop the 'name' column for prediction (model only needs features)
features = prediction_data.drop(columns=['name'])

# Fill any remaining NaN values with median (safety check)
if features.isnull().any().any():
    print("   Note: Filling missing values with column medians...")
    features = features.fillna(features.median())

# Scale the features using the SAME scaler from training
print("4. Scaling features with training scaler...")
features_scaled = scaler.transform(features)  # Use transform, NOT fit_transform!

# Make predictions
print("5. Making predictions...")
predictions = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)

# Create results DataFrame
results = pd.DataFrame({
    'Name': names,
    'Prediction': predictions,
    'Status': ['Healthy' if p == 0 else 'Parkinson\'s Detected' for p in predictions],
    'Confidence_Healthy': [f"{prob[0]*100:.1f}%" for prob in probabilities],
    'Confidence_Parkinsons': [f"{prob[1]*100:.1f}%" for prob in probabilities]
})

# Display results
print("\n" + "=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)
print(results.to_string(index=False))
print("=" * 70)

# Summary statistics
healthy_count = (predictions == 0).sum()
pd_count = (predictions == 1).sum()

print(f"\nSUMMARY:")
print(f"  • Predicted Healthy: {healthy_count} people")
print(f"  • Predicted Parkinson's: {pd_count} people")

# Save results to CSV
output_file = 'data/prediction_results.csv'
results.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print("• Status 0 (Healthy): Voice patterns similar to healthy individuals")
print("• Status 1 (Parkinson's): Voice patterns show characteristics")
print("  associated with Parkinson's Disease")
print("\nNOTE: This is a screening tool, NOT a medical diagnosis.")
print("Anyone with Parkinson's prediction should consult a medical professional.")
print("=" * 70)
