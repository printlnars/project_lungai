import joblib
import pandas as pd
import numpy as np
import random

# Загрузка модели и колонок
model = joblib.load("best_model.pkl")
sample_df = pd.read_csv("dataset_final.csv", nrows=1)
expected_cols = sample_df.columns.tolist()
if "LUNG_CANCER" in expected_cols:
    expected_cols.remove("LUNG_CANCER")

# Уникальные города из датасета
cities = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань"]

def get_risk(data):
    # Предобработка DataFrame
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
    
    # 1. Возрастной вклад
    age_score = min(max((age - 30) / 50.0, 0.0), 1.0) * 0.15
    
    # 2. Вклад курения
    smoke_score = min(pack_years / 40.0, 1.0) * 0.25 if smoker else 0.0
    
    # 3. Вклад симптомов
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
    
    # 4. Анамнез
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
    x_points = [0.0, 0.00300009, 0.05400012, 0.10800344, 0.13800350, 0.19924515, 0.72874359, 0.94898411, 1.0]
    y_points = [0.01, 0.0115, 0.0696, 0.1970, 0.2608, 0.4660, 0.8548, 0.9840, 0.99]
    
    final_proba = float(np.interp(combined_score, x_points, y_points))
    final_proba = min(max(final_proba, 0.01), 0.99)
    
    return final_proba

def run_monotonicity_test(n_patients=1000):
    print(f"Запуск автоматической проверки монотонности системы на {n_patients} пациентах...")
    
    binary_violations = 0
    continuous_violations = 0
    total_checks = 0
    
    binary_features = [
        'hemoptysis', 'shortness_of_breath', 'voice_change', 'weakness', 
        'cough', 'swallowing_problems', 'chest_pain', 'arm_shoulder_pain', 
        'dry_cough', 'weight_loss', 'appetite_loss', 'family_cancer_history', 
        'pulmonologist_followup', 'smoker'
    ]
    
    for i in range(n_patients):
        # Генерируем случайного пациента
        age = random.randint(18, 90)
        smoker = random.choice([0, 1])
        smoking_years = random.randint(1, age - 15) if smoker else 0
        cigs_per_day = random.randint(1, 60) if smoker else 0
        arvi = random.randint(0, 10)
        
        patient = {
            "age": age,
            "sex": random.choice([0, 1]),
            "birth_place": random.choice(cities),
            "residence": random.choice(cities),
            "smoker": smoker,
            "smoking_years": smoking_years,
            "cigs_per_day": cigs_per_day,
            "arvi_per_year": arvi
        }
        
        for feat in binary_features:
            if feat != 'smoker':
                patient[feat] = random.choice([0, 1])
        
        # 1. Проверяем бинарные симптомы и признаки
        for feat in binary_features:
            if feat == 'smoker':
                continue # smoker проверяем отдельно с непрерывными
                
            # Копируем и ставим 0
            p0 = patient.copy()
            p0[feat] = 0
            
            # Копируем и ставим 1
            p1 = patient.copy()
            p1[feat] = 1
            
            r0 = get_risk(p0)
            r1 = get_risk(p1)
            
            total_checks += 1
            if r1 < r0:
                print(f"Ошибка монотонности (бинарный признак {feat}): r(1)={r1:.4f} < r(0)={r0:.4f}")
                binary_violations += 1
                
        # 2. Проверяем непрерывные признаки
        # А. Возраст (+5 лет)
        pa0 = patient.copy()
        pa1 = patient.copy()
        pa1['age'] = min(pa1['age'] + 5, 100)
        
        ra0 = get_risk(pa0)
        ra1 = get_risk(pa1)
        total_checks += 1
        if ra1 < ra0:
            print(f"Ошибка монотонности (возраст): r(age+5)={ra1:.4f} < r(age)={ra0:.4f}")
            continuous_violations += 1
            
        # Б. ОРВИ (+2 в год)
        pv0 = patient.copy()
        pv1 = patient.copy()
        pv1['arvi_per_year'] = min(pv1['arvi_per_year'] + 2, 12)
        
        rv0 = get_risk(pv0)
        rv1 = get_risk(pv1)
        total_checks += 1
        if rv1 < rv0:
            print(f"Ошибка монотонности (ОРВИ): r(arvi+2)={rv1:.4f} < r(arvi)={rv0:.4f}")
            continuous_violations += 1
            
        # В. Курение (если курильщик, стаж +5 лет)
        if patient['smoker'] == 1:
            ps0 = patient.copy()
            ps1 = patient.copy()
            ps1['smoking_years'] = ps1['smoking_years'] + 5
            
            rs0 = get_risk(ps0)
            rs1 = get_risk(ps1)
            total_checks += 1
            if rs1 < rs0:
                print(f"Ошибка монотонности (стаж курения): r(years+5)={rs1:.4f} < r(years)={rs0:.4f}")
                continuous_violations += 1
                
            # Сигареты (+10 штук в день)
            pc0 = patient.copy()
            pc1 = patient.copy()
            pc1['cigs_per_day'] = pc1['cigs_per_day'] + 10
            
            rc0 = get_risk(pc0)
            rc1 = get_risk(pc1)
            total_checks += 1
            if rc1 < rc0:
                print(f"Ошибка монотонности (сигареты в день): r(cigs+10)={rc1:.4f} < r(cigs)={rc0:.4f}")
                continuous_violations += 1
                
            # Переключение с некурящего на курящего
            pn0 = patient.copy()
            pn0['smoker'] = 0
            pn0['smoking_years'] = 0
            pn0['cigs_per_day'] = 0
            
            pn1 = patient.copy() # курящий с теми же параметрами
            
            rn0 = get_risk(pn0)
            rn1 = get_risk(pn1)
            total_checks += 1
            if rn1 < rn0:
                print(f"Ошибка монотонности (статус курильщика): r(smoker=1)={rn1:.4f} < r(smoker=0)={rn0:.4f}")
                continuous_violations += 1

    print(f"\n--- Результаты тестирования ---")
    print(f"Всего проверок: {total_checks}")
    print(f"Нарушений бинарной монотонности: {binary_violations}")
    print(f"Нарушений непрерывной монотонности: {continuous_violations}")
    print(f"Итог: {'УСПЕХ' if (binary_violations + continuous_violations == 0) else 'ОШИБКА'}")

if __name__ == "__main__":
    run_monotonicity_test()
