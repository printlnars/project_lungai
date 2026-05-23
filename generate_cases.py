import pandas as pd
import numpy as np
import random

def generate_synthetic_cancer_cases(n=800):
    # Загружаем структуру колонок из существующего файла
    df_existing = pd.read_csv('dataset_final.csv')
    # Убираем старые сгенерированные данные, оставляем только оригинальные
    df_existing = df_existing[df_existing['LUNG_CANCER'] == 0]
    cols = df_existing.columns.tolist()
    
    new_rows = []
    
    cities = df_existing['birth_place'].unique().tolist()
    
    for _ in range(n):
        row = {}
        
        # Генерируем "более реалистичного" больного раком легких
        row['age'] = random.randint(45, 85)
        row['sex'] = random.choice([0, 1])
        row['birth_place'] = random.choice(cities)
        row['residence'] = random.choice(cities)
        
        # Курение - фактор риска, выражен сильнее у больных
        is_heavy_smoker = random.random() < 0.85
        if is_heavy_smoker:
            row['smoker'] = 1
            row['smoking_years'] = random.randint(30, 65)
            row['cigs_per_day'] = random.randint(20, 45)
        else:
            row['smoker'] = 0
            row['smoking_years'] = 0
            row['cigs_per_day'] = 0
            
        row['arvi_per_year'] = random.randint(3, 7)
        row['family_cancer_history'] = 1 if random.random() < 0.50 else 0
        
        # Симптомы у больных раком должны быть чаще, чем у здоровых (класс 0), но с пересечением
        row['hemoptysis'] = 1 if random.random() < 0.45 else 0 # > 0.09
        row['shortness_of_breath'] = 1 if random.random() < 0.75 else 0 # > 0.64
        row['voice_change'] = 1 if random.random() < 0.45 else 0 # > 0.29
        row['weakness'] = 1 if random.random() < 0.80 else 0 # > 0.67
        row['cough'] = 1 if random.random() < 0.75 else 0 # > 0.58
        row['swallowing_problems'] = 1 if random.random() < 0.60 else 0 # > 0.47
        row['chest_pain'] = 1 if random.random() < 0.70 else 0 # > 0.55
        row['arm_shoulder_pain'] = 1 if random.random() < 0.40 else 0 # > 0.25
        row['dry_cough'] = 1 if random.random() < 0.55 else 0 # > 0.39
        row['weight_loss'] = 1 if random.random() < 0.45 else 0 # > 0.18
        row['appetite_loss'] = 1 if random.random() < 0.45 else 0 # > 0.19
        row['pulmonologist_followup'] = 1 if random.random() < 0.40 else 0 # > 0.09
        
        row['LUNG_CANCER'] = 1
        
        new_rows.append(row)
    
    df_new = pd.DataFrame(new_rows)
    # Гарантируем порядок колонок
    df_new = df_new[cols]
    
    # Объединяем
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Сохраняем
    df_final.to_csv('dataset_final.csv', index=False)
    print(f"[OK] Successfully added {n} realistic cancer cases to database.")
    print(f"Total rows: {len(df_final)}")
    print(f"Class distribution: {df_final['LUNG_CANCER'].value_counts().to_dict()}")

if __name__ == "__main__":
    generate_synthetic_cancer_cases()
