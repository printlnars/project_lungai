import pandas as pd
df = pd.read_csv("dataset_final.csv")
print("arvi_per_year describe for Class 0:")
print(df[df['LUNG_CANCER'] == 0]['arvi_per_year'].describe())
print("\narvi_per_year describe for Class 1:")
print(df[df['LUNG_CANCER'] == 1]['arvi_per_year'].describe())
