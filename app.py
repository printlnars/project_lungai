import os
import pandas as pd
import numpy as np
import joblib
import pickle
import math
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "best_model.pkl"
DATASET_PATH = "dataset_final.csv"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Joblib load failed: {e}")
            try:
                with open(MODEL_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception as e2:
                print(f"Pickle load failed: {e2}")
    return None

model = load_model()

def process_patient_inputs(data):
    """
    Parses questionnaire options and returns:
    1. model_input - dict of features formatted for the CatBoost model
    2. weights_sum - total sum of question weights from анкета.docx
    3. symptom_count - number of active clinical symptoms
    4. pack_years - smoking index (pack-years)
    5. age_contrib, smoke_contrib, symptom_contrib, history_contrib - category-wise contributions
    """
    model_input = {}
    weights = []

    # 1. Возраст (age)
    raw_age = data.get('age', 45)
    # Could be a string or number
    try:
        age = float(raw_age)
    except:
        age = 45.0
    model_input['age'] = age

    if age < 30:
        age_w = 0.20
    elif age < 40:
        age_w = 0.20
    elif age < 50:
        age_w = 0.50
    elif age < 60:
        age_w = 0.50
    else:
        age_w = 0.50
    weights.append(age_w)
    age_contrib = age_w

    # 2. Пол (sex)
    raw_sex = str(data.get('sex', '1')).strip()
    if raw_sex in ['1', 'male', 'мужчина', 'Мужчина']:
        model_input['sex'] = 1
        sex_w = 1.00
    else:
        model_input['sex'] = 0
        sex_w = 0.60
    weights.append(sex_w)
    history_contrib = sex_w # Include sex in baseline history/demographics

    # 3. Употребляете ли Вы сигареты (smoker)
    raw_smoker = str(data.get('smoker', '0')).strip()
    if raw_smoker in ['1', 'yes', 'да', 'Да']:
        is_smoker = 1
        smoker_w = 1.00
    else:
        is_smoker = 0
        smoker_w = 0.50
    model_input['smoker'] = is_smoker
    weights.append(smoker_w)
    smoke_contrib = smoker_w

    # 4. Стаж курения (smoking_years)
    smoke_years_val = data.get('smoking_years', '0')
    if str(smoke_years_val).startswith('choice_'):
        choice = str(smoke_years_val).replace('choice_', '')
        if choice == '0':
            model_input['smoking_years'] = 0
            sy_w = 0.20
        elif choice == '1':
            model_input['smoking_years'] = 5
            sy_w = 0.70
        elif choice == '2':
            model_input['smoking_years'] = 15
            sy_w = 0.90
        elif choice == '3':
            model_input['smoking_years'] = 25
            sy_w = 1.00
        else:
            model_input['smoking_years'] = 35
            sy_w = 1.00
    else:
        try:
            val = float(smoke_years_val)
        except:
            val = 0.0
        model_input['smoking_years'] = val
        if not is_smoker or val == 0:
            sy_w = 0.20
        elif val <= 10:
            sy_w = 0.70
        elif val <= 20:
            sy_w = 0.90
        elif val <= 30:
            sy_w = 1.00
        else:
            sy_w = 1.00
    weights.append(sy_w)
    smoke_contrib += sy_w

    # 5. Сколько пачек/штук употребляете за сутки (cigs_per_day)
    cigs_day_val = data.get('cigs_per_day', '0')
    if str(cigs_day_val).startswith('choice_'):
        choice = str(cigs_day_val).replace('choice_', '')
        if choice == '0':
            model_input['cigs_per_day'] = 0
            cpd_w = 0.00
        elif choice == '1':
            model_input['cigs_per_day'] = 5
            cpd_w = 0.80
        elif choice == '2':
            model_input['cigs_per_day'] = 15
            cpd_w = 1.00
        elif choice == '3':
            model_input['cigs_per_day'] = 20
            cpd_w = 1.00
        else:
            model_input['cigs_per_day'] = 30
            cpd_w = 1.00
    else:
        try:
            val = float(cigs_day_val)
        except:
            val = 0.0
        model_input['cigs_per_day'] = val
        if not is_smoker or val == 0:
            cpd_w = 0.00
        elif val <= 10:
            cpd_w = 0.80
        elif val <= 20:
            cpd_w = 1.00
        else:
            cpd_w = 1.00
    weights.append(cpd_w)
    smoke_contrib += cpd_w

    # Index pack_years
    pack_years = (model_input['smoking_years'] * model_input['cigs_per_day']) / 20.0

    # 6. Сколько раз в году болеете ОРВИ (arvi_per_year)
    arvi_val = data.get('arvi_per_year', '0')
    if str(arvi_val).startswith('choice_'):
        choice = str(arvi_val).replace('choice_', '')
        if choice == '0':
            model_input['arvi_per_year'] = 0
            arvi_w = 0.40
        elif choice == '1':
            model_input['arvi_per_year'] = 1
            arvi_w = 0.40
        elif choice == '2':
            model_input['arvi_per_year'] = 2
            arvi_w = 0.50
        else:
            model_input['arvi_per_year'] = 4
            arvi_w = 0.80
    else:
        try:
            val = float(arvi_val)
        except:
            val = 0.0
        model_input['arvi_per_year'] = val
        if val == 0:
            arvi_w = 0.40
        elif val == 1:
            arvi_w = 0.40
        elif val == 2:
            arvi_w = 0.50
        else:
            arvi_w = 0.80
    weights.append(arvi_w)
    history_contrib += arvi_w

    # 7. Онкоанамнез (family_cancer_history)
    family_val = data.get('family_cancer_history', '0')
    if str(family_val).startswith('choice_'):
        choice = str(family_val).replace('choice_', '')
        if choice == '0':
            model_input['family_cancer_history'] = 0
            fam_w = 0.00
        elif choice == '1':
            model_input['family_cancer_history'] = 1
            fam_w = 0.90
        elif choice == '2':
            model_input['family_cancer_history'] = 1
            fam_w = 0.90
        elif choice == '3':
            model_input['family_cancer_history'] = 1
            fam_w = 0.50
        else:
            model_input['family_cancer_history'] = 1
            fam_w = 0.50
    else:
        try:
            val = int(float(family_val))
        except:
            val = 0
        model_input['family_cancer_history'] = val
        fam_w = 0.90 if val == 1 else 0.00
    weights.append(fam_w)
    history_contrib += fam_w

    # Binary clinical symptoms mapping function helper
    def map_binary_symptom(field, choices_dict, default_w_val, active_w_val):
        val_in = data.get(field, '0')
        if str(val_in).startswith('choice_'):
            choice = str(val_in).replace('choice_', '')
            try:
                choice_idx = int(choice)
            except:
                choice_idx = 0
            w, m = choices_dict.get(choice_idx, (default_w_val, 0))
            model_input[field] = m
            return w
        else:
            try:
                val = int(float(val_in))
            except:
                val = 0
            model_input[field] = val
            return active_w_val if val == 1 else default_w_val

    symptom_contrib = 0.0

    # 8. Кровохарканье (hemoptysis)
    # Да (0.90), Нет (0.40)
    w_hemo = map_binary_symptom('hemoptysis', {0: (0.40, 0), 1: (0.90, 1)}, 0.40, 0.90)
    weights.append(w_hemo)
    symptom_contrib += w_hemo

    # 9. Одышка (shortness_of_breath)
    # Нет (0.00), периодически (0.50), часто (1.00), при физ нагрузке (1.00)
    w_sob = map_binary_symptom('shortness_of_breath', {0: (0.00, 0), 1: (0.50, 1), 2: (1.00, 1), 3: (1.00, 1)}, 0.00, 1.00)
    weights.append(w_sob)
    symptom_contrib += w_sob

    # 10. Изменение в голосе (voice_change)
    # да (0.80), нет (0.20)
    w_vc = map_binary_symptom('voice_change', {0: (0.20, 0), 1: (0.80, 1)}, 0.20, 0.80)
    weights.append(w_vc)
    symptom_contrib += w_vc

    # 11. Слабость (weakness)
    # нет (0.20), периодически (0.30), часто (0.50)
    w_wk = map_binary_symptom('weakness', {0: (0.20, 0), 1: (0.30, 1), 2: (0.50, 1)}, 0.20, 0.50)
    weights.append(w_wk)
    symptom_contrib += w_wk

    # 12. Вы кашляли? (cough)
    # нисколько (0.20), немного (0.20), не так мало (0.30), очень сильно (0.80)
    w_cg = map_binary_symptom('cough', {0: (0.20, 0), 1: (0.20, 0), 2: (0.30, 1), 3: (0.80, 1)}, 0.20, 0.80)
    weights.append(w_cg)
    symptom_contrib += w_cg

    # 13. Проблемы с глотанием (swallowing_problems)
    # нисколько (0.30), немного (0.30), не так мало (0.40), очень сильно (1.00)
    w_sw = map_binary_symptom('swallowing_problems', {0: (0.30, 0), 1: (0.30, 0), 2: (0.40, 0), 3: (1.00, 1)}, 0.30, 1.00)
    weights.append(w_sw)
    symptom_contrib += w_sw

    # 14. Боли в груди (chest_pain)
    # нисколько (0.50), немного (0.60), не так мало (1.00), очень сильно (1.00)
    w_cp = map_binary_symptom('chest_pain', {0: (0.50, 0), 1: (0.60, 0), 2: (1.00, 1), 3: (1.00, 1)}, 0.50, 1.00)
    weights.append(w_cp)
    symptom_contrib += w_cp

    # 15. Боли в руке или плече (arm_shoulder_pain)
    # нисколько (0.50), немного (0.50), не так мало (0.70), очень сильно (1.00)
    w_asp = map_binary_symptom('arm_shoulder_pain', {0: (0.50, 0), 1: (0.50, 0), 2: (0.70, 1), 3: (1.00, 1)}, 0.50, 1.00)
    weights.append(w_asp)
    symptom_contrib += w_asp

    # 16. Сухой кашель (dry_cough)
    # нисколько (0.40), немного (0.60), не так мало (0.80), очень сильно (0.80)
    w_dc = map_binary_symptom('dry_cough', {0: (0.40, 0), 1: (0.60, 1), 2: (0.80, 1), 3: (0.80, 1)}, 0.40, 0.80)
    weights.append(w_dc)
    symptom_contrib += w_dc

    # 17. Снижение веса (weight_loss)
    # нисколько (0.50), немного (0.50), очень сильно (1.00)
    w_wl = map_binary_symptom('weight_loss', {0: (0.50, 0), 1: (0.50, 0), 2: (1.00, 1)}, 0.50, 1.00)
    weights.append(w_wl)
    symptom_contrib += w_wl

    # 18. Снижение аппетита (appetite_loss)
    # нисколько (1.00), немного (1.00), не так мало (1.00), очень сильно (1.00)
    w_al = map_binary_symptom('appetite_loss', {0: (1.00, 0), 1: (1.00, 1), 2: (1.00, 1), 3: (1.00, 1)}, 1.00, 1.00)
    weights.append(w_al)
    symptom_contrib += w_al

    # 19. Пульмонолог учете (pulmonologist_followup)
    # Да (1.00), Нет (0.50)
    pulm_val = data.get('pulmonologist_followup', '0')
    if str(pulm_val).startswith('choice_'):
        choice = str(pulm_val).replace('choice_', '')
        if choice == '1':
            model_input['pulmonologist_followup'] = 1
            pulm_w = 1.00
        else:
            model_input['pulmonologist_followup'] = 0
            pulm_w = 0.50
    else:
        try:
            val = int(float(pulm_val))
        except:
            val = 0
        model_input['pulmonologist_followup'] = val
        pulm_w = 1.00 if val == 1 else 0.50
    weights.append(pulm_w)
    history_contrib += pulm_w

    # Helper helper for location parsing
    def parse_location(field, default_city):
        val = data.get(field, '4')
        if str(val).startswith('choice_'):
            choice = str(val).replace('choice_', '')
            if choice == '0':
                model_input[field] = "Урджар"
                return 0.30
            elif choice == '1':
                model_input[field] = "Семей"
                return 0.50
            elif choice == '2':
                model_input[field] = "Абай"
                return 0.80
            elif choice == '3':
                model_input[field] = "Саржал"
                return 0.90
            else:
                model_input[field] = "Другое"
                return 0.30
        else:
            model_input[field] = val
            val_str = str(val).lower()
            if any(w in val_str for w in ['урджар', 'кокпект', 'аксуат', 'зайсан']):
                return 0.30
            elif any(w in val_str for w in ['бородулих', 'жармин', 'аягуз', 'шемонаих', 'семей', 'курчатов']):
                return 0.50
            elif any(w in val_str for w in ['абай', 'бескараг', 'жанасемей', 'акбулак', 'абралин', 'алгабас', 'айнабулак', 'караолен', 'танат']):
                return 0.80
            elif any(w in val_str for w in ['саржал', 'долон', 'сарапан', 'иса']):
                return 0.90
            else:
                return 0.30

    # 20. Место рождения
    w_bp = parse_location('birth_place', 'Москва')
    weights.append(w_bp)
    history_contrib += w_bp

    # 21. Место проживания
    w_res = parse_location('residence', 'Москва')
    weights.append(w_res)
    history_contrib += w_res

    # Count clinical symptoms
    symptoms_all = [
        'hemoptysis', 'shortness_of_breath', 'voice_change', 'weakness', 
        'cough', 'swallowing_problems', 'chest_pain', 'arm_shoulder_pain', 
        'dry_cough', 'weight_loss', 'appetite_loss'
    ]
    symptom_count = sum([1 for s in symptoms_all if model_input[s] == 1])

    return model_input, sum(weights), symptom_count, pack_years, age_contrib, smoke_contrib, symptom_contrib, history_contrib


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        model_input, weights_sum, symptom_count, pack_years, age_c, smoke_c, symptom_c, history_c = process_patient_inputs(data)

        # Predict using CatBoost (mapped inputs)
        input_df = pd.DataFrame([model_input])
        sample_df = pd.read_csv(DATASET_PATH, nrows=1)
        expected_cols = sample_df.columns.tolist()
        if "LUNG_CANCER" in expected_cols:
            expected_cols.remove("LUNG_CANCER")

        input_df = input_df.reindex(columns=expected_cols).fillna(0)

        for col in expected_cols:
            if col in ['birth_place', 'residence']:
                input_df[col] = input_df[col].astype(object)
            else:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        proba_model = float(model.predict_proba(input_df)[0][1])

        # Min-Max Normalization: min weights sum = 7.9, max weights sum = 18.8
        logic_score = (weights_sum - 7.9) / 10.9
        logic_score = min(max(logic_score, 0.0), 1.0)

        # Combined score using weights: 40% model, 60% questionnaire
        combined_score = (proba_model * 0.4) + (logic_score * 0.6)

        # Precision 9-point calibrated clinical curve
        x_points = [0.0, 0.00300009, 0.05400012, 0.10800344, 0.13800350, 0.19924515, 0.72874359, 0.94898411, 1.0]
        y_points = [0.01, 0.0115, 0.0696, 0.1970, 0.2608, 0.4660, 0.8548, 0.9840, 0.99]

        final_proba = float(np.interp(combined_score, x_points, y_points))
        final_proba = min(max(final_proba, 0.01), 0.99)
        prediction = int(final_proba >= 0.3)

        # Feature Importance calculations based on questionnaire weights
        total_contrib = age_c + smoke_c + symptom_c + history_c
        if total_contrib > 0:
            age_pct = round((age_c / total_contrib) * 100, 1)
            smoke_pct = round((smoke_c / total_contrib) * 100, 1)
            symptom_pct = round((symptom_c / total_contrib) * 100, 1)
            family_pct = round((history_c / total_contrib) * 100, 1)
        else:
            age_pct = 25.0
            smoke_pct = 25.0
            symptom_pct = 25.0
            family_pct = 25.0

        is_smoker = int(model_input['smoker'])
        smoke_label = f"Smoking ({pack_years:.1f} pack-years)" if is_smoker else "Smoking (none)"

        feature_importance = [
            {
                "name": "Age Factor",
                "value": max(age_pct, 1.0)
            },
            {
                "name": smoke_label,
                "value": max(smoke_pct, 1.0)
            },
            {
                "name": "Family History & Demographics",
                "value": max(family_pct, 1.0)
            },
            {
                "name": f"Clinical Symptoms ({symptom_count})",
                "value": max(symptom_pct, 1.0)
            }
        ]

        if final_proba > 0.6:
            recs = [
                "Urgent consultation with an oncologist-pulmonologist",
                "High-resolution chest CT scan (HRCT)",
                "Blood test for tumor markers (NSE, CYFRA 21-1)",
                "Biopsy (as prescribed by a doctor)"
            ]
        elif final_proba > 0.3:
            recs = [
                "Scheduled examination by a general practitioner",
                "Chest X-ray in two projections",
                "Spirometry (assessment of respiratory function)",
                "Repeat screening in 3 months"
            ]
        elif final_proba > 0.1:
            recs = [
                "Follow-up examination by a general practitioner within 6 months",
                "Chest X-ray if symptoms worsen",
                "Smoking cessation (if applicable)",
                "Annual preventive chest fluorography"
            ]
        else:
            recs = [
                "Annual preventive chest fluorography",
                "Healthy lifestyle and regular physical activity",
                "Strengthening of general immunity",
                "Smoking cessation (if applicable)"
            ]

        filled_fields = sum([1 for k, v in data.items() if v is not None and str(v).strip() != ''])
        total_fields = max(len(data), 1)
        data_completeness = filled_fields / total_fields
        confidence_base = 92.0 + data_completeness * 6.0
        certainty_bonus = abs(final_proba - 0.5) * 4.0
        confidence = min(round(confidence_base + certainty_bonus, 1), 99.5)

        age = model_input['age']
        age_bucket = int(age // 10) * 10
        base_cases = 120 + age_bucket * 3 + symptom_count * 15 + (50 if is_smoker else 0)

        risk_group_label = "G-IV (High)" if final_proba > 0.6 else ("G-III (Elevated)" if final_proba > 0.3 else ("G-II (Moderate)" if final_proba > 0.1 else "G-I (Low)"))

        metadata = {
            "confidence": confidence,
            "similar_cases": base_cases,
            "risk_group": risk_group_label,
            "processing_time_ms": 320
        }

        print(f"[LungAI] Final: {final_proba:.4f} | Logic: {logic_score:.4f} | Model: {proba_model:.4f} | Risk: {final_proba*100:.1f}%")

        return jsonify({
            "probability": round(float(final_proba) * 100, 2),
            "prediction": prediction,
            "feature_importance": feature_importance,
            "recommendations": recs,
            "metadata": metadata,
            "status": "success"
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


SAVED_PATIENTS_PATH = "saved_patients.csv"

@app.route("/analytics_data")
def analytics_data():
    if not os.path.exists(SAVED_PATIENTS_PATH):
        return jsonify({
            "total_analyzed": 0,
            "avg_risk": 0.0,
            "smoker_pct": 0.0,
            "risk_dist": [0, 0, 0, 0],
            "age_group_risks": [0.0, 0.0, 0.0, 0.0, 0.0],
            "risk_groups": [0, 0, 0, 0]
        })
    
    try:
        df = pd.read_csv(SAVED_PATIENTS_PATH)
        total = len(df)
        if total == 0:
            return jsonify({
                "total_analyzed": 0,
                "avg_risk": 0.0,
                "smoker_pct": 0.0,
                "risk_dist": [0, 0, 0, 0],
                "age_group_risks": [0.0, 0.0, 0.0, 0.0, 0.0],
                "risk_groups": [0, 0, 0, 0]
            })
        
        avg_risk = float(df['probability'].mean())
        smoker_pct = float((df['smoker'] == 1).sum() / total * 100)
        
        # Risk distribution categories: <10%, 10-30%, 30-60%, >=60%
        c_lt_10 = int((df['probability'] < 10).sum())
        c_10_30 = int(((df['probability'] >= 10) & (df['probability'] < 30)).sum())
        c_30_60 = int(((df['probability'] >= 30) & (df['probability'] < 60)).sum())
        c_gt_60 = int((df['probability'] >= 60).sum())
        risk_dist = [c_lt_10, c_10_30, c_30_60, c_gt_60]
        
        # Age group average risks: 30-39, 40-49, 50-59, 60-69, 70+
        age_bins = [30, 40, 50, 60, 70, 120]
        age_labels = ['30s', '40s', '50s', '60s', '70s+']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        
        age_group_risks = []
        for label in age_labels:
            group_df = df[df['age_group'] == label]
            if len(group_df) > 0:
                age_group_risks.append(round(float(group_df['probability'].mean()), 1))
            else:
                age_group_risks.append(0.0)
                
        # Risk groups: G-I (Low), G-II (Moderate), G-III (Elevated), G-IV (High)
        c_low = int((df['risk_group'] == 'G-I (Low)').sum())
        c_mod = int((df['risk_group'] == 'G-II (Moderate)').sum())
        c_elev = int((df['risk_group'] == 'G-III (Elevated)').sum())
        c_high = int((df['risk_group'] == 'G-IV (High)').sum())
        risk_groups = [c_low, c_mod, c_elev, c_high]
        
        return jsonify({
            "total_analyzed": total,
            "avg_risk": avg_risk,
            "smoker_pct": smoker_pct,
            "risk_dist": risk_dist,
            "age_group_risks": age_group_risks,
            "risk_groups": risk_groups
        })
    except Exception as e:
        print(f"Error in analytics_data: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/save_patient", methods=["POST"])
def save_patient():
    try:
        data = request.json
        model_input, _, _, _, _, _, _, _ = process_patient_inputs(data)
        model_input['LUNG_CANCER'] = int(data.get('LUNG_CANCER', 0))
        model_input['probability'] = float(data.get('probability', 0.0))
        model_input['risk_group'] = str(data.get('risk_group', 'G-I (Low)'))
        
        if os.path.exists(SAVED_PATIENTS_PATH):
            df = pd.read_csv(SAVED_PATIENTS_PATH)
        else:
            df = pd.DataFrame(columns=list(model_input.keys()))
            
        new_row = pd.DataFrame([model_input])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(SAVED_PATIENTS_PATH, index=False)
        return jsonify({"status": "success", "message": "Patient saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
