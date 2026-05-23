# 🫁 LungAI Pro — Advanced AI-Powered Lung Cancer Diagnostics

LungAI Pro is a clinically validated, state-of-the-art web application designed for early-stage lung cancer risk assessment. The system utilizes a strictly monotonic machine learning pipeline powered by **CatBoostClassifier**, blended with professional clinical decision rules and a 9-point precision calibration curve.

All calculations and predictions are performed **locally on the server**, ensuring absolute patient data privacy and high performance.

---

## ✨ Key Features
* 🛡️ **Perfect Monotonicity:** Risk predictions are mathematically guaranteed to be strictly monotonic (e.g., increasing age, smoking years, or active clinical symptoms will *never* decrease the calculated risk score).
* 🎯 **Precision Calibration:** Utilizes a custom 9-point clinical calibration curve (`np.interp`) to blend ML predictions with expert clinical rules, preventing abrupt risk jumps and ensuring highly realistic probability distributions.
* 📊 **Interactive Analytics:** Features modern, beautiful charts (using Chart.js) depicting factor importance, risk distributions, daily diagnostics activity, and risk groups.
* 💎 **Glassmorphism UI:** A stunning, fully responsive "glass-style" user interface tailored for both desktop and mobile devices.
* 🔒 **Local & Private:** Zero external API calls. Patient records are stored locally and securely.

---

## 🚀 Step-by-Step Launch Instructions

To launch the project on your machine, you must have **Python 3.10 or higher** installed.

### Step 1: Clone or Unpack the Project
Clone the repository to your local folder:
```bash
git clone https://github.com/printlnars/project_lungai.git
cd project_lungai
```
*(Or simply extract the project ZIP archive to any directory and open your terminal in that folder).*

### Step 2: Set Up Virtual Environment
Create and activate a clean Python virtual environment to manage dependencies:

* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On Windows (CMD):**
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Required Dependencies
Upgrade `pip` and install all required machine learning and web dependencies listed in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Application Server
Start the local Flask server by running `app.py`:
```bash
python app.py
```

### Step 5: Open the Diagnostic Web Interface
Once the terminal displays `* Running on http://127.0.0.1:5000`, open any modern browser and navigate to:

👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📂 Project Structure & Key Files

* 📄 `app.py` — The core Flask application server containing strictly monotonic risk scoring, factor analysis, and API endpoints.
* 📂 `templates/index.html` — The premium web interface structure, styles loading, dynamic particle canvas, and layouts.
* 📂 `static/` — Assets folder:
  * `css/style.css` — High-end styling rules, gradients, and custom responsive layouts.
  * `js/main.js` — Client-side interaction logic, async server calls, loader animations, and Chart.js integrations.
* 📦 `best_model.pkl` — The trained `CatBoostClassifier` pipeline (includes categorical preprocessing and one-hot encoding).
* 📊 `dataset_final.csv` — The anonymized training dataset (contains 4,800 patients, including balanced synthetic cases).
* 📄 `requirements.txt` — List of all required Python modules and dependencies.
* 🧪 `test_cases_app.py` — Automated verification script testing 7 distinct clinical diagnostic scenarios.

---

## ⚕️ Clinical Disclaimer

> [!WARNING]  
> **Important:** LungAI Pro is an educational and auxiliary screening tool designed to assist in risk evaluation. Its results are for informational purposes only and **do not substitute** for professional medical consultation, diagnostics, or treatment. Any medical decisions must be made by a licensed oncologist or pulmonologist based on clinical examinations (CT scan, bronchoscopy, histology, etc.).
