import joblib
import pandas as pd
import numpy as np

model = joblib.load("best_model.pkl")
df = pd.read_csv("dataset_final.csv")
print("Total rows:", len(df))
print("Class 1 rows:", len(df[df['LUNG_CANCER'] == 1]))

# Let's predict on some random rows of Class 1
X_cancer = df[df['LUNG_CANCER'] == 1].drop(columns=['LUNG_CANCER'])
proba_cancer = model.predict_proba(X_cancer)[:, 1]
print("Cancer class 1 model probas (mean, std, min, max):", 
      np.mean(proba_cancer), np.std(proba_cancer), np.min(proba_cancer), np.max(proba_cancer))

X_healthy = df[df['LUNG_CANCER'] == 0].drop(columns=['LUNG_CANCER'])
proba_healthy = model.predict_proba(X_healthy)[:, 1]
print("Healthy class 0 model probas (mean, std, min, max):", 
      np.mean(proba_healthy), np.std(proba_healthy), np.min(proba_healthy), np.max(proba_healthy))
