import joblib, pandas as pd, numpy as np, random

model = joblib.load('best_model.pkl')
sample_df = pd.read_csv('dataset_final.csv', nrows=1)
expected_cols = [c for c in sample_df.columns if c != 'LUNG_CANCER']
cities = ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Казань']

def get_risk(data):
    input_df = pd.DataFrame([data]).reindex(columns=expected_cols).fillna(0)
    for col in expected_cols:
        if col in ['birth_place', 'residence']:
            input_df[col] = input_df[col].astype(object)
        else:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
    pm = float(model.predict_proba(input_df)[0][1])
    age = float(data.get('age', 0))
    smoker = int(data.get('smoker', 0))
    pack_years = (float(data.get('smoking_years', 0)) * float(data.get('cigs_per_day', 0))) / 20.0
    age_score = min(max((age - 30) / 50.0, 0.0), 1.0) * 0.15
    smoke_score = min(pack_years / 40.0, 1.0) * 0.25 if smoker else 0.0
    sw = {
        'hemoptysis': 0.09, 'weight_loss': 0.08, 'appetite_loss': 0.06,
        'shortness_of_breath': 0.05, 'cough': 0.05, 'dry_cough': 0.05,
        'chest_pain': 0.05, 'weakness': 0.04, 'voice_change': 0.04,
        'swallowing_problems': 0.04, 'arm_shoulder_pain': 0.03
    }
    symp = sum(sw.get(s, 0) for s in sw if int(data.get(s, 0)) == 1)
    h = 0.0
    if int(data.get('family_cancer_history', 0)):
        h += 0.07
    if int(data.get('pulmonologist_followup', 0)):
        h += 0.05
    h += min(float(data.get('arvi_per_year', 0)) / 6.0, 1.0) * 0.03
    logic = age_score + smoke_score + symp + h
    comb = pm * 0.4 + logic * 0.6
    x_pts = [0.0, 0.00300009, 0.05400012, 0.10800344, 0.13800350, 0.19924515, 0.72874359, 0.94898411, 1.0]
    y_pts = [0.01, 0.0115, 0.0696, 0.197, 0.2608, 0.466, 0.8548, 0.984, 0.99]
    return float(np.interp(comb, x_pts, y_pts))

binary_features = [
    'hemoptysis', 'shortness_of_breath', 'voice_change', 'weakness',
    'cough', 'swallowing_problems', 'chest_pain', 'arm_shoulder_pain',
    'dry_cough', 'weight_loss', 'appetite_loss',
    'family_cancer_history', 'pulmonologist_followup'
]

bviol = 0
cviol = 0
total = 0
random.seed(42)

print("Запуск проверки монотонности на 200 пациентах...")

for i in range(200):
    smoker = random.choice([0, 1])
    p = {
        'age': random.randint(18, 90),
        'sex': random.choice([0, 1]),
        'birth_place': random.choice(cities),
        'residence': random.choice(cities),
        'smoker': smoker,
        'smoking_years': random.randint(1, 40) if smoker else 0,
        'cigs_per_day': random.randint(1, 40) if smoker else 0,
        'arvi_per_year': random.randint(0, 10)
    }
    for f in binary_features:
        p[f] = random.choice([0, 1])

    # Бинарные признаки
    for feat in binary_features:
        p0 = p.copy(); p0[feat] = 0
        p1 = p.copy(); p1[feat] = 1
        r0 = get_risk(p0); r1 = get_risk(p1)
        total += 1
        if r1 < r0 - 1e-9:
            bviol += 1
            print(f"  BINARY VIOLATION {feat}: r(0)={r0:.4f} > r(1)={r1:.4f}")

    # Возраст +10 лет
    pa1 = p.copy(); pa1['age'] = min(p['age'] + 10, 95)
    total += 1
    if get_risk(pa1) < get_risk(p) - 1e-9:
        cviol += 1
        print(f"  AGE violation: p={i}")

    # ОРВИ +3
    pv1 = p.copy(); pv1['arvi_per_year'] = min(p['arvi_per_year'] + 3, 12)
    total += 1
    if get_risk(pv1) < get_risk(p) - 1e-9:
        cviol += 1
        print(f"  ARVI violation: p={i}")

    if smoker:
        # Стаж курения +10 лет
        ps1 = p.copy(); ps1['smoking_years'] = p['smoking_years'] + 10
        total += 1
        if get_risk(ps1) < get_risk(p) - 1e-9:
            cviol += 1
            print(f"  SMOKE_YEARS violation: p={i}")

        # Сигареты +10
        pc1 = p.copy(); pc1['cigs_per_day'] = p['cigs_per_day'] + 10
        total += 1
        if get_risk(pc1) < get_risk(p) - 1e-9:
            cviol += 1
            print(f"  CIGS violation: p={i}")

print()
print("--- Результаты ---")
print(f"Всего проверок:                 {total}")
print(f"Нарушений бинарной монотонности:    {bviol}")
print(f"Нарушений непрерывной монотонности: {cviol}")
if bviol + cviol == 0:
    print("Итог: PASS — нарушений нет!")
else:
    print("Итог: FAIL — обнаружены нарушения!")
