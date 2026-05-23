import sys
sys.stdout.reconfigure(encoding='utf-8')

with open("train_output.txt", "r", encoding="utf-16") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "CatBoostClassifier" in line:
        print(f"--- Found on line {i+1} ---")
        for j in range(max(0, i-5), min(len(lines), i+15)):
            print(f"{j+1}: {lines[j].strip()}")
