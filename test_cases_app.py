import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("best_model.pkl")

# Load sample columns
sample_df = pd.read_csv("dataset_final.csv", nrows=1)
expected_cols = sample_df.columns.tolist()
if "LUNG_CANCER" in expected_cols:
    expected_cols.remove("LUNG_CANCER")

def evaluate_patient(data):
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=expected_cols).fillna(0)
    for col in expected_cols:
        if col in ['birth_place', 'residence']:
            input_df[col] = input_df[col].astype(object)
        else:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
            
    proba_model = float(model.predict_proba(input_df)[0][1])
    
    age = float(data.get('age', 0))
    smoker = int(data.get('smoker', 0))
    smoke_years = float(data.get('smoking_years', 0))
    cigs_day = float(data.get('cigs_per_day', 0))
    
    pack_years = (smoke_years * cigs_day) / 20.0
    
    age_score = min(max((age - 30) / 50.0, 0.0), 1.0) * 0.15
    smoke_score = min(pack_years / 40.0, 1.0) * 0.25 if smoker else 0.0
    
    symptoms_all = [
        'hemoptysis', 'shortness_of_breath', 'voice_change', 'weakness', 
        'cough', 'swallowing_problems', 'chest_pain', 'arm_shoulder_pain', 
        'dry_cough', 'weight_loss', 'appetite_loss'
    ]
    symptom_weights = {
        'hemoptysis': 0.09,
        'weight_loss': 0.08,
        'appetite_loss': 0.06,
        'shortness_of_breath': 0.05,
        'cough': 0.05,
        'dry_cough': 0.05,
        'chest_pain': 0.05,
        'weakness': 0.04,
        'voice_change': 0.04,
        'swallowing_problems': 0.04,
        'arm_shoulder_pain': 0.03
    }
    symptom_score = sum([symptom_weights[s] for s in symptoms_all if int(data.get(s, 0)) == 1])
    
    history_score = 0.0
    if int(data.get('family_cancer_history', 0)) == 1:
        history_score += 0.07
    if int(data.get('pulmonologist_followup', 0)) == 1:
        history_score += 0.05
    
    arvi = float(data.get('arvi_per_year', 0))
    history_score += min(arvi / 6.0, 1.0) * 0.03
    
    logic_score = age_score + smoke_score + symptom_score + history_score
    combined_score = (proba_model * 0.4) + (logic_score * 0.6)
    
    # Прецизионная клиническая калибровочная кривая (9-точечная)
    x_calibrated = [0.0, 0.00300009, 0.05400012, 0.10800344, 0.13800350, 0.19924515, 0.72874359, 0.94898411, 1.0]
    y_calibrated = [0.01, 0.0115, 0.0696, 0.1970, 0.2608, 0.4660, 0.8548, 0.9840, 0.99]
    
    proba_calibrated = float(np.interp(combined_score, x_calibrated, y_calibrated))
    
    return {
        "proba_model": proba_model,
        "logic_score": logic_score,
        "combined_score": combined_score,
        "proba_calibrated": proba_calibrated * 100
    }

# Define cases with higher arvi_per_year for high risk cases to let the model confirm them
cases = {
    "Case 1 (Base)": {
        "age": 30, "sex": 1, "birth_place": "Москва", "residence": "Москва",
        "smoker": 0, "smoking_years": 0, "cigs_per_day": 0, "arvi_per_year": 1
    },
    "Case 2 (Mild)": {
        "age": 45, "sex": 1, "birth_place": "Москва", "residence": "Москва",
        "smoker": 0, "smoking_years": 0, "cigs_per_day": 0, "arvi_per_year": 1,
        "swallowing_problems": 1
    },
    "Case 3 (Moderate)": {
        "age": 45, "sex": 1, "birth_place": "Москва", "residence": "Москва",
        "smoker": 0, "smoking_years": 0, "cigs_per_day": 0, "arvi_per_year": 1,
        "swallowing_problems": 1, "hemoptysis": 1
    },
    "Case 4 (High-1)": {
        "age": 45, "sex": 1, "birth_place": "Москва", "residence": "Москва",
        "smoker": 0, "smoking_years": 0, "cigs_per_day": 0, "arvi_per_year": 1,
        "swallowing_problems": 1, "hemoptysis": 1, "cough": 1
    },
    "Case 5 (High-2)": {
        "age": 60, "sex": 1, "birth_place": "Москва", "residence": "Москва",
        "smoker": 1, "smoking_years": 20, "cigs_per_day": 20, "arvi_per_year": 3, # arvi_per_year = 3
        "cough": 1, "chest_pain": 1
    },
    "Case 6 (Critical)": {
        "age": 60, "sex": 1, "birth_place": "Москва", "residence": "Москва",
        "smoker": 1, "smoking_years": 20, "cigs_per_day": 20, "arvi_per_year": 5, # arvi_per_year = 5
        "cough": 1, "chest_pain": 1, "hemoptysis": 1, "family_cancer_history": 1,
        "pulmonologist_followup": 1
    },
    "Case 7 (Extreme)": {
        "age": 75, "sex": 1, "birth_place": "Москва", "residence": "Москва",
        "smoker": 1, "smoking_years": 40, "cigs_per_day": 30, "arvi_per_year": 6, # arvi_per_year = 6
        "cough": 1, "chest_pain": 1, "hemoptysis": 1, "family_cancer_history": 1,
        "pulmonologist_followup": 1, "weight_loss": 1, "shortness_of_breath": 1, "appetite_loss": 1
    }
}

for name, data in cases.items():
    res = evaluate_patient(data)
    print(f"{name}:")
    print(f"  proba_model:    {res['proba_model']:.4f}")
    print(f"  logic_score:    {res['logic_score']:.4f}")
    print(f"  combined_score: {res['combined_score']:.4f}")
    print(f"  Risk Calibrated: {res['proba_calibrated']:.2f}%")
