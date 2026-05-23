import joblib
import pandas as pd
import numpy as np

model = joblib.load("best_model.pkl")
df = pd.read_csv("dataset_final.csv")
expected_cols = df.columns.tolist()
if "LUNG_CANCER" in expected_cols:
    expected_cols.remove("LUNG_CANCER")

# Let's inspect some actual rows from dataset_final.csv where LUNG_CANCER == 1
cancer_df = df[df['LUNG_CANCER'] == 1].head(5)
for idx, row in cancer_df.iterrows():
    print(f"\n--- Cancer Row {idx} ---")
    for col in expected_cols:
        val = row[col]
        if col in ['age', 'smoking_years', 'cigs_per_day', 'arvi_per_year']:
            print(f"  {col}: {val}")
        elif val == 1:
            print(f"  {col}: 1")
    # Predict
    input_df = pd.DataFrame([row.drop('LUNG_CANCER')])
    proba = model.predict_proba(input_df)[0][1]
    print(f"  --> MODEL PROBA: {proba:.4f}")
