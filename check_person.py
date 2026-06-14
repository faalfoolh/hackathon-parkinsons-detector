#!/usr/bin/env python3
"""
Parkinson's Disease Detection - Person Checker
Clean output for presentation
"""

import pandas as pd
import joblib
import numpy as np
import sys
import warnings

# Suppress sklearn warnings for clean output
warnings.filterwarnings('ignore')

def check_person(person_name):
    """Check if a person has Parkinson's Disease"""

    # Load all data
    df = pd.read_csv('data/voice_features_combined_all.csv')
    model = joblib.load('model/final_trained_model.pkl')
    scaler = joblib.load('model/final_scaler.pkl')

    # Find the person (case-insensitive, partial match)
    person_name_lower = person_name.lower()
    matches = df[df['name'].str.lower().str.contains(person_name_lower)]

    if len(matches) == 0:
        print("\n" + "="*70)
        print(f"❌ PERSON NOT FOUND: '{person_name}'")
        print("="*70)
        print("\nAvailable people:")
        for name in df['name'].head(15):
            print(f"  • {name}")
        return

    if len(matches) > 1:
        print("\n" + "="*70)
        print(f"⚠️  MULTIPLE MATCHES FOUND FOR: '{person_name}'")
        print("="*70)
        for name in matches['name']:
            print(f"  • {name}")
        print("\nPlease be more specific.")
        return

    # Get the person's data
    person = matches.iloc[0]

    # Prepare features for prediction
    feature_cols = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
                   'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                   'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
                   'NHR', 'HNR']

    X = person[feature_cols].values.reshape(1, -1)
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    X_scaled = scaler.transform(X)

    # Get prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    # Display results
    print("\n" + "="*70)
    print("PARKINSON'S DISEASE DETECTION RESULT")
    print("="*70)

    print(f"\n📋 PERSON: {person['name']}")
    print(f"   Actual Status: {person['actual_status']}")

    print("\n" + "-"*70)

    if prediction == 1:
        print("🔴 DIAGNOSIS: PARKINSON'S DISEASE DETECTED")
        print(f"   Confidence: {probability[1]*100:.1f}%")
    else:
        print("✅ DIAGNOSIS: HEALTHY")
        print(f"   Confidence: {probability[0]*100:.1f}%")

    print("-"*70)

    print("\n📊 VOICE BIOMARKERS:")
    print("-"*70)

    # Show key features in a clean format
    print(f"  Fundamental Frequency (Fo):  {person['MDVP:Fo(Hz)']:8.2f} Hz")
    print(f"  Maximum Frequency (Fhi):     {person['MDVP:Fhi(Hz)']:8.2f} Hz")
    print(f"  Minimum Frequency (Flo):     {person['MDVP:Flo(Hz)']:8.2f} Hz")
    print(f"\n  Jitter (% variation):        {person['MDVP:Jitter(%)']:8.4f} %")
    print(f"  Shimmer (amplitude var):     {person['MDVP:Shimmer']:8.4f}")
    print(f"\n  HNR (harmonics/noise):       {person['HNR']:8.2f} dB")
    print(f"  NHR (noise/harmonics):       {person['NHR']:8.4f}")

    print("\n" + "="*70)

    # Show risk level
    risk_pct = probability[1] * 100
    if risk_pct < 10:
        risk_level = "VERY LOW RISK"
    elif risk_pct < 30:
        risk_level = "LOW RISK"
    elif risk_pct < 50:
        risk_level = "MODERATE RISK"
    elif risk_pct < 70:
        risk_level = "HIGH RISK"
    else:
        risk_level = "VERY HIGH RISK"

    print(f"Risk Assessment: {risk_level} ({risk_pct:.1f}%)")
    print("="*70)
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line argument
        person_name = " ".join(sys.argv[1:])
        check_person(person_name)
    else:
        # Interactive mode
        print("\n" + "="*70)
        print("PARKINSON'S DISEASE DETECTION SYSTEM")
        print("="*70)
        print("\nEnter person's name (or 'quit' to exit)")

        while True:
            print("\n" + "-"*70)
            person_name = input("Enter name: ").strip()

            if person_name.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if person_name:
                check_person(person_name)
