import joblib
import pandas as pd

pipeline = joblib.load("best_model.pkl")
print("Pipeline steps:")
for step_name, step_obj in pipeline.steps:
    print(f"  {step_name}: {type(step_obj)}")
    if hasattr(step_obj, "estimator"):
        print(f"    estimator: {type(step_obj.estimator)}")
    if step_name == "classifier" or step_name == "model":
        if hasattr(step_obj, "get_params"):
            print("Classifier params:")
            params = step_obj.get_params()
            for k, v in params.items():
                if k in ['monotone_constraints', 'depth', 'iterations', 'learning_rate', 'n_estimators', 'max_depth']:
                    print(f"      {k}: {v}")
