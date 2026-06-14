# feature_plotter.py - Phase 4: Feature Visualization

import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = 'parkinsons/parkinsons.csv' 
FEATURE_A = 'MDVP:Jitter(%)'
FEATURE_B = 'MDVP:Shimmer'

print("--- Starting Phase 4: Generating Feature Comparison Plots ---")

try:
    # Load the raw data again (handling the header issue)
    df = pd.read_csv(FILE_PATH, header=None)
    df.columns = df.iloc[0]
    df = df[1:].copy()
    df['status'] = pd.to_numeric(df['status'])
    
    # Ensure features are numeric for plotting
    df[FEATURE_A] = pd.to_numeric(df[FEATURE_A])
    df[FEATURE_B] = pd.to_numeric(df[FEATURE_B])
    
except Exception as e:
    print(f"\nERROR: Could not load data for plotting. Details: {e}")
    exit()

# Filter data into two groups (Healthy and PD)
healthy = df[df['status'] == 0]
pd_group = df[df['status'] == 1]

# --- Plot 1: Jitter Comparison ---
plt.figure(figsize=(8, 6))
plt.boxplot([healthy[FEATURE_A], pd_group[FEATURE_A]], 
            labels=['Healthy (0)', 'Parkinson\'s (1)'])
plt.title(f'Distribution of {FEATURE_A} by Patient Status')
plt.ylabel(FEATURE_A)
plt.grid(axis='y', alpha=0.5)
print(f"Plot 1: Box Plot for {FEATURE_A} is ready.")
plt.show()

# --- Plot 2: Shimmer Comparison ---
plt.figure(figsize=(8, 6))
plt.boxplot([healthy[FEATURE_B], pd_group[FEATURE_B]], 
            labels=['Healthy (0)', 'Parkinson\'s (1)'])
plt.title(f'Distribution of {FEATURE_B} by Patient Status')
plt.ylabel(FEATURE_B)
plt.grid(axis='y', alpha=0.5)
print(f"Plot 2: Box Plot for {FEATURE_B} is ready.")
plt.show()

print("\n--- Phase 4 COMPLETE: Two plots generated. Use them for your presentation! ---")