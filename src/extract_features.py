import pandas as pd
import numpy as np
import os

DATA = "data"

# --- FILE PATHS (Using original file names for direct access) ---
# Assuming these files are in the directory where you run the script.
USER_DATA_PATH = DATA + '/voice_features_database.csv'
PARKINSONS_TEMPLATE_PATH = DATA + '/parkinsons.csv'
OUTPUT_GROUND_TRUTH_FILE = DATA + '/voice_features_database_GROUND_TRUTH.csv'
OUTPUT_PREDICTION_INPUT_FILE = DATA + '/voice_features_database_PREDICTION_INPUT.csv'
# ------------------------------------------------------------------

try:
    # 1. Load both CSV files
    user_df = pd.read_csv(USER_DATA_PATH)
    parkinsons_df = pd.read_csv(PARKINSONS_TEMPLATE_PATH)
    
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Files not found. Ensure '{USER_DATA_PATH}' and '{PARKINSONS_TEMPLATE_PATH}' are in the same folder as this script. Error: {e}")
    exit()

# 2. Get the full list of 24 target column names from the template
target_columns = parkinsons_df.columns.tolist()

# 3. Prepare user data (Remove 'Label' and align 'name' column)
df_final = user_df.drop(columns=['Label'], errors='ignore')
if 'Name' in df_final.columns:
    df_final = df_final.rename(columns={'Name': 'name'})

# 4. === HANDLE COLUMN STRUCTURE ===
# Check if data already has all 24 columns or needs positional renaming
if len(df_final.columns) == 19:
    # Case: Data has only 19 columns (name + 18 features)
    # Map the user's 19 columns to the first 19 names of the template
    new_names = target_columns[:19]
    df_final.columns = new_names
elif len(df_final.columns) == 24:
    # Case: Data already has all 24 columns with correct headers
    # No renaming needed, columns already match template
    pass
else:
    raise ValueError(f"Unexpected number of columns: {len(df_final.columns)}. Expected 19 or 24.")

# 5. === SMART IMPUTATION FOR REMAINING COLUMNS ===
# Impute missing columns: RPDE, DFA, spread1, spread2, D2, PPE
# Filter the template data to get statistics from ONLY healthy patients (status=0)
healthy_stats_df = parkinsons_df[parkinsons_df['status'] == 0]
num_rows_to_fill = len(df_final)

# Determine which columns need imputation
# Include RPDE (index 18) and columns 19-23
cols_to_impute = target_columns[18:19] + target_columns[19:]  # RPDE + last 5 columns
if len(df_final.columns) == 24:
    # For 24-column data, only impute columns that are empty/NaN
    cols_to_impute = [col for col in cols_to_impute if df_final[col].isna().all()]

for col in cols_to_impute:
    # Get mean and standard deviation for this column from healthy patients
    mean = healthy_stats_df[col].mean()
    std = healthy_stats_df[col].std()

    # Generate random numbers based on that mean and std
    random_values = np.random.normal(loc=mean, scale=std, size=num_rows_to_fill)
    df_final[col] = random_values

# Explicitly set status to 0 for all rows (we don't know if patients have Parkinson's)
df_final['status'] = 0

# 6. === CREATE GROUND TRUTH FILE (The Answer Key) ===
# Re-order all 24 columns to exactly match the template and set the known status (0)
df_ground_truth = df_final[target_columns].copy()
df_ground_truth['status'] = 0 

# Save File 1
df_ground_truth.to_csv(OUTPUT_GROUND_TRUTH_FILE, index=False)


# 7. === CREATE PREDICTION INPUT FILE (The Test File) ===
# Take the Ground Truth file (which has the correct order) and drop the 'status' column.
df_prediction_input = df_ground_truth.drop(columns=['status'])

# Save File 2
df_prediction_input.to_csv(OUTPUT_PREDICTION_INPUT_FILE, index=False)

print("\n------------------------------------------------------------")
print(f"SUCCESS: Two files have been prepared for testing:")
print(f"1. Test Input: '{OUTPUT_PREDICTION_INPUT_FILE}' (No status column)")
print(f"2. Ground Truth: '{OUTPUT_GROUND_TRUTH_FILE}' (Status column = 0)")
print("------------------------------------------------------------")