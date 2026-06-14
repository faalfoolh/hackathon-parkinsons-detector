# pd_detector.py - Phase 1: Load, Scale, and Split Data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION (UPDATED FILE PATH) ---
# This path now tells Python to look INSIDE the 'data' folder
FILE_PATH = 'data/parkinsons.csv'
# -----------------------------------------

print("--- Starting Phase 1: Data Preparation ---")

# ----------------------------------------------------
# STEP 1: Load Data and Handle UCI Formatting
# ----------------------------------------------------
try:
    # Load the data using header=None to bypass the UCI file formatting issue
    df = pd.read_csv(FILE_PATH, header=None)
    
    # Set column names and drop the header row
    df.columns = df.iloc[0]
    df = df[1:].copy()
    
    # Convert the 'status' column (your target) to a number (0 or 1)
    df['status'] = pd.to_numeric(df['status'])
    # Drop the 'name' column (the patient ID)
    df = df.drop(['name'], axis=1)

except FileNotFoundError:
    print(f"\nFATAL ERROR: File Not Found. Check if the file is at: {FILE_PATH}")
    exit()
except Exception as e:
    print(f"\nFATAL ERROR: Data processing issue. Details: {e}")
    exit()

X = df.drop(['status'], axis=1) # Features (all the voice measurements)
y = df['status'] # Target (0 or 1)
print(f"Data loaded successfully! Total samples: {len(df)}")


# ----------------------------------------------------
# STEP 2: Scaling (Normalization)
# ----------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("Features scaled successfully!")

# Save the scaler for later use in predictions
import joblib
import os
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved to 'model/scaler.pkl'")


# ----------------------------------------------------
# STEP 3: Train/Test Split (The Handoff)
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42 
)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- Handoff Point: Save the split data as CSV files ---
# These four files are the output for the rest of your team!
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("\n--- Phase 1 COMPLETE: Saved X_train, X_test, y_train, y_test as CSV files! ---")