import joblib
import sys

# Set standard output encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

pipeline = joblib.load("best_model.pkl")
preprocessor = pipeline.named_steps["preprocess"]
if "cat" in preprocessor.named_transformers_:
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    print("Categories for birth_place:")
    for c in onehot.categories_[0]:
        print(c)
    print("\nCategories for residence:")
    for c in onehot.categories_[1]:
        print(c)
else:
    print("No categorical transformer found in preprocessor")
