import joblib
import pickle
import os

files = ["best_model.pkl", "model.pkl", "catboost_model.pkl", "lightgbm_model.pkl"]

for f_name in files:
    if not os.path.exists(f_name):
        continue
    print(f"--- Checking {f_name} ---")
    
    # Try pickle
    try:
        with open(f_name, "rb") as f:
            obj = pickle.load(f)
        print(f"  [Pickle] Success! Type: {type(obj)}")
        continue
    except Exception as e:
        print(f"  [Pickle] Failed: {e}")
    
    # Try joblib
    try:
        obj = joblib.load(f_name)
        print(f"  [Joblib] Success! Type: {type(obj)}")
        continue
    except Exception as e:
        print(f"  [Joblib] Failed: {e}")

    # Try catboost
    try:
        import catboost as cb
        model = cb.CatBoostClassifier()
        model.load_model(f_name)
        print(f"  [CatBoost] Success!")
        continue
    except Exception as e:
        print(f"  [CatBoost] Failed: {e}")
