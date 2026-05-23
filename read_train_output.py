import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    with open("train_output.txt", "r", encoding="utf-16") as f:
        for i in range(50):
            line = f.readline()
            if not line:
                break
            print(line.strip())
except Exception as e:
    print("UTF-16 failed, trying UTF-8:", e)
    with open("train_output.txt", "r", encoding="utf-8") as f:
        for i in range(50):
            line = f.readline()
            if not line:
                break
            print(line.strip())
