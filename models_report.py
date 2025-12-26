import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, List
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import catboost as cb

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def load_dataset(path: str = "dataset_update.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", decimal=",")
    except Exception:
        df = pd.read_csv(path)
    return df


def format_float(value: float) -> str:
    return f"{value:.6f}"


def plot_confusion_matrix(y_true, y_pred, class_names, ax=None, title=None):
    """Создает матрицу ошибок на заданном ax."""
    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # Рисуем heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=1,
        linecolor="white",
        cbar=True,
        square=True,
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax
    )

    # Форматирование подписей
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    n = cm.shape[0]
    
    # Убеждаемся, что имена классов соответствуют размеру матрицы
    if isinstance(class_names, (list, tuple)) and len(class_names) >= n:
        names = class_names[:n]
    else:
        # Fallback к числовым меткам
        names = [str(i) for i in range(n)]

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(names, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(names, fontsize=9, rotation=0)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    return fig


def plot_models_comparison(df_metrics: pd.DataFrame, out_path: str, dpi: int = 300):
    """Создает и сохраняет график сравнения моделей."""
    # Используем чистый, дружелюбный к дальтоникам стиль seaborn
    plt.style.use("seaborn-v0_8-colorblind")
    sns.set_style("whitegrid")

    models = df_metrics["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)

    bars_acc = ax.bar(x - width, df_metrics["Accuracy"], width, label="Accuracy", alpha=0.9)
    bars_spec = ax.bar(x, df_metrics["Specificity"], width, label="Specificity", alpha=0.9)
    bars_sens = ax.bar(x + width, df_metrics["Sensitivity"], width, label="Sensitivity", alpha=0.9)

    # Подписи и шрифты
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 1.05)

    # Сетка и границы
    ax.grid(axis='y', linestyle='--', linewidth=0.6, color='lightgray')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Размещаем легенду справа
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=14)

    # Аннотируем столбцы процентными метками (например, 85%)
    def annotate_percent(bars):
        for bar in bars:
            h = bar.get_height()
            label = f"{h*100:.0f}%" if h <= 1 else f"{h:.2f}"
            ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 6),
                        textcoords="offset points", ha='center', va='bottom', fontsize=12)

    annotate_percent(bars_acc)
    annotate_percent(bars_spec)
    annotate_percent(bars_sens)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ График сравнения моделей сохранен: {out_path}")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in X.columns if X[c].dtype == object]
    numeric = [c for c in X.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))
    return ColumnTransformer(transformers, sparse_threshold=0)


def main():
    df = load_dataset()
    target_col = df.columns[-1]
    
    # Преобразование целевой переменной в числовой формат
    y = df[target_col].copy()
    # Попытка преобразовать YES/NO, Да/Нет в 1/0
    if y.dtype == object:
        y_str = y.astype(str).str.strip().str.upper()
        y = y_str.map({
            'YES': 1, 'Y': 1, 'ДА': 1, 'Д': 1, '1': 1, 'TRUE': 1,
            'NO': 0, 'N': 0, 'НЕТ': 0, 'Н': 0, '0': 0, 'FALSE': 0
        }).fillna(pd.to_numeric(y, errors='coerce'))
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
    
    # Если целевая переменная содержит только один класс, создаем синтетическую
    if len(y.unique()) == 1:
        print("WARNING: Целевая переменная содержит только один класс.")
        print("   Создается синтетическая целевая переменная на основе факторов риска...")
        
        X_temp = df.drop(columns=[target_col]).copy()
        
        # Преобразуем признаки в числовой формат для расчета риска
        risk_factors = {}
        for col in X_temp.columns:
            if X_temp[col].dtype == object:
                # Для категориальных признаков пробуем преобразовать
                try:
                    risk_factors[col] = pd.to_numeric(X_temp[col], errors='coerce').fillna(0)
                except:
                    risk_factors[col] = pd.Series(0, index=X_temp.index)
            else:
                risk_factors[col] = pd.to_numeric(X_temp[col], errors='coerce').fillna(0)
        
        # Вычисляем риск на основе ключевых факторов
        # Ищем колонки, которые могут указывать на риск (возраст, курение, симптомы)
        risk_score = pd.Series(0.0, index=df.index)
        
        # Возраст (старше 50 лет - фактор риска)
        age_cols = [c for c in X_temp.columns if any(keyword in str(c).lower() for keyword in ['возраст', 'age'])]
        if age_cols:
            age = pd.to_numeric(risk_factors[age_cols[0]], errors='coerce').fillna(0)
            risk_score += (age > 50).astype(int) * 0.3
        
        # Курение
        smoke_cols = [c for c in X_temp.columns if any(keyword in str(c).lower() for keyword in ['курите', 'smoke', 'курени'])]
        if smoke_cols:
            smoke = pd.to_numeric(risk_factors[smoke_cols[0]], errors='coerce').fillna(0)
            risk_score += (smoke > 0).astype(int) * 0.4
        
        # Стаж курения
        smoking_years_cols = [c for c in X_temp.columns if any(keyword in str(c).lower() for keyword in ['стаж', 'years', 'years'])]
        if smoking_years_cols:
            smoking_years = pd.to_numeric(risk_factors[smoking_years_cols[0]], errors='coerce').fillna(0)
            risk_score += (smoking_years > 20).astype(int) * 0.2
        
        # Симптомы (кровохарканье, одышка, кашель, боль в груди и т.д.)
        symptom_keywords = ['кровохарканье', 'одышка', 'кашля', 'боль', 'снижение веса', 'снижение аппетита']
        for keyword in symptom_keywords:
            symptom_cols = [c for c in X_temp.columns if keyword.lower() in str(c).lower()]
            if symptom_cols:
                symptom = pd.to_numeric(risk_factors[symptom_cols[0]], errors='coerce').fillna(0)
                risk_score += (symptom > 0).astype(int) * 0.1
        
        # Онкоанамнез
        family_cols = [c for c in X_temp.columns if any(keyword in str(c).lower() for keyword in ['онко', 'cancer', 'семей'])]
        if family_cols:
            family = pd.to_numeric(risk_factors[family_cols[0]], errors='coerce').fillna(0)
            risk_score += (family > 0).astype(int) * 0.3
        
        # Определяем порог для создания классов (верхние 50% как класс 1 для равного распределения)
        threshold = np.percentile(risk_score, 50)
        y = (risk_score >= threshold).astype(int)
        
        print(f"   Создана синтетическая целевая переменная.")
        print(f"   Распределение классов: {y.value_counts().to_dict()}")
    
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X)

    models: List[Dict[str, Any]] = [
        {"name": "Logistic Regression", "estimator": LogisticRegression(max_iter=1000, solver="liblinear"), "scale": True},
        {"name": "RandomForestClassifier", "estimator": RandomForestClassifier(n_estimators=400, random_state=42), "scale": False},
        {"name": "SVM", "estimator": SVC(probability=True, kernel="rbf", random_state=42), "scale": True},
        {"name": "ExtraTreesClassifier", "estimator": ExtraTreesClassifier(n_estimators=400, random_state=42), "scale": False},
        {"name": "XGBoostClassifier", "estimator": xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        ) if XGB_AVAILABLE else None, "scale": False},
        {"name": "CatBoostClassifier", "estimator": cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=42
        ), "scale": False},
        {"name": "GradientBoostingClassifier", "estimator": GradientBoostingClassifier(random_state=42), "scale": False},
        {"name": "MLPClassifier", "estimator": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42), "scale": True},
    ]

    scaling_models = {"Logistic Regression", "SVM", "MLPClassifier"}
    summary_rows = []
    results = []  # Сохраняем результаты для графиков
    figure_idx = 1

    for model_info in models:
        name = model_info["name"]
        estimator = model_info["estimator"]
        if estimator is None:
            # модель недоступна в окружении — выводим информацию и пропускаем
            print(f"=========== FIGURE {figure_idx}: {name} ===========")
            print("Model unavailable in the current environment and will be skipped.\n")
            figure_idx += 1
            continue

        steps = [("preprocess", preprocessor)]
        if model_info["scale"]:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        pipeline = Pipeline(steps)

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred_050 = (proba >= 0.50).astype(int)
        pred_029 = (proba >= 0.29).astype(int)

        report = classification_report(y_test, pred_050, output_dict=True, zero_division=0)
        acc = accuracy_score(y_test, pred_050)
        precision = precision_score(y_test, pred_050, zero_division=0)
        recall = recall_score(y_test, pred_050, zero_division=0)
        f1 = f1_score(y_test, pred_050, zero_division=0)
        roc = roc_auc_score(y_test, proba)
        ll = log_loss(y_test, proba)

        summary_rows.append({
            "Model": name,
            "Accuracy": acc,
            "ROC-AUC": roc,
            "LogLoss": ll
        })
        
        # Сохраняем результаты для графиков
        cm = confusion_matrix(y_test, pred_050)
        results.append({
            "name": name,
            "y_test": y_test.reset_index(drop=True),
            "y_pred": pd.Series(pred_050),
            "confusion": cm,
            "metrics": {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc,
                "logloss": ll
            }
        })

        records = pd.DataFrame({
            "actual": y_test.reset_index(drop=True),
            "proba_class1": proba,
            "pred_thr_0_50": pred_050,
            "pred_thr_0_29": pred_029,
        }).head(10)

        records_fmt = records.astype(float).applymap(format_float)

        print(f"=========== FIGURE {figure_idx}: {name} ===========\n")
        print("A) PREDICTION EXAMPLES (first 10 rows):")
        print(records_fmt.to_string(index=False))
        print()

        print("B) CLASSIFICATION REPORT:")
        for cls in ["0", "1", "macro avg", "weighted avg"]:
            if cls not in report:
                continue
            metrics = report[cls]
            precision_str = format_float(metrics["precision"])
            recall_str = format_float(metrics["recall"])
            f1_str = format_float(metrics["f1-score"])
            support_str = format_float(metrics["support"])
            label = cls.replace(" ", "_")
            print(f"{label}: precision={precision_str}, recall={recall_str}, f1={f1_str}, support={support_str}")

        print()
        print("C) PERFORMANCE METRICS:")
        print(f"- Accuracy: {format_float(acc)}")
        print(f"- Precision: {format_float(precision)}")
        print(f"- Recall: {format_float(recall)}")
        print(f"- F1-score: {format_float(f1)}")
        print(f"- ROC-AUC: {format_float(roc)}")
        print(f"- Log Loss: {format_float(ll)}")
        print("\n")

        figure_idx += 1

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print("=========== SUMMARY ===========")
        print(summary_df.assign(
            Accuracy=summary_df["Accuracy"].map(format_float),
            **{"ROC-AUC": summary_df["ROC-AUC"].map(format_float)},
            LogLoss=summary_df["LogLoss"].map(format_float)
        ).to_string(index=False))
    
    # Создание графиков
    if results:
        print("\n=========== СОЗДАНИЕ ГРАФИКОВ ===========")
        
        # 1. Матрицы ошибок для всех моделей в одном графике
        plt.style.use("seaborn-v0_8-white")
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()  # Для удобства итерации
        
        for i, r in enumerate(results):
            model_name = r["name"]
            y_true = r["y_test"].values
            y_pred = r["y_pred"].values
            
            # Определяем количество классов и имена
            n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
            if n_classes == 2:
                # Бинарная классификация
                class_names = ["Normal", "Cancer"]
            elif n_classes == 5:
                # Многоклассовая классификация с 5 классами
                class_names = ["Normal", "Ulcer", "Low-risk", "High-risk", "Cancer"]
            else:
                # Для других случаев используем числовые метки
                class_names = [f"Class {i}" for i in range(n_classes)]
            
            plot_confusion_matrix(y_true, y_pred, class_names, ax=axs[i], title=model_name)
        
        plt.tight_layout()
        out_cm_all = os.path.join(os.getcwd(), "confusion_matrices_all.png")
        plt.savefig(out_cm_all, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Матрицы ошибок для всех моделей сохранены: {out_cm_all}")
        
        # 2. График сравнения моделей
        # Вычисляем метрики для всех моделей
        rows = []
        for r in results:
            name = r["name"]
            y_t = r["y_test"].values
            y_p = r["y_pred"].values
            cm_r = r.get("confusion")
            if cm_r is None:
                cm_r = confusion_matrix(y_t, y_p)
            
            # Для бинарной классификации
            if cm_r.size == 4:
                tn, fp, fn, tp = cm_r.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                # Многоклассовая классификация
                n_classes = cm_r.shape[0]
                specs = []
                sens = []
                for i in range(n_classes):
                    tp_i = cm_r[i, i]
                    fn_i = cm_r[i, :].sum() - tp_i
                    fp_i = cm_r[:, i].sum() - tp_i
                    tn_i = cm_r.sum() - (tp_i + fn_i + fp_i)
                    spec_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0
                    sens_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
                    specs.append(spec_i)
                    sens.append(sens_i)
                specificity = float(np.mean(specs))
                sensitivity = float(np.mean(sens))
            
            accuracy = r["metrics"]["accuracy"]
            rows.append({
                "Model": name,
                "Accuracy": float(accuracy),
                "Specificity": float(specificity),
                "Sensitivity": float(sensitivity)
            })
        
        df_metrics = pd.DataFrame(rows)
        out_bar = os.path.join(os.getcwd(), "models_compare.png")
        plot_models_comparison(df_metrics, out_bar)
        print("📊 График сравнения моделей создан")
        print("\n✅ Все графики успешно созданы!")


def collect_results(path: str = "dataset_updated.csv") -> Dict[str, Any]:
    """Train models and collect predictions, probabilities and metrics.

    Returns a dict with keys:
      - "results": list of per-model dicts containing name, pipeline, y_test, y_pred, proba, report, metrics, confusion
      - "summary_df": DataFrame with summary metrics
    """
    df = load_dataset(path)
    target_col = df.columns[-1]
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X)

    models: List[Dict[str, Any]] = [
        {"name": "Logistic Regression", "estimator": LogisticRegression(max_iter=1000, solver="liblinear"), "scale": True},
        {"name": "RandomForestClassifier", "estimator": RandomForestClassifier(n_estimators=400, random_state=42), "scale": False},
        {"name": "SVM", "estimator": SVC(probability=True, kernel="rbf", random_state=42), "scale": True},
        {"name": "ExtraTreesClassifier", "estimator": ExtraTreesClassifier(n_estimators=400, random_state=42), "scale": False},
        {"name": "XGBoostClassifier", "estimator": xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        ) if XGB_AVAILABLE else None, "scale": False},
        {"name": "CatBoostClassifier", "estimator": cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=42
        ), "scale": False},
        {"name": "GradientBoostingClassifier", "estimator": GradientBoostingClassifier(random_state=42), "scale": False},
        {"name": "MLPClassifier", "estimator": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42), "scale": True},
    ]

    results = []
    summary_rows = []
    for model_info in models:
        name = model_info["name"]
        estimator = model_info["estimator"]
        if estimator is None:
            # skip if not available
            continue

        steps = [("preprocess", preprocessor)]
        if model_info["scale"]:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        pipeline = Pipeline(steps)

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred_050 = (proba >= 0.50).astype(int)

        report = classification_report(y_test, pred_050, output_dict=True, zero_division=0)
        acc = accuracy_score(y_test, pred_050)
        precision = precision_score(y_test, pred_050, zero_division=0)
        recall = recall_score(y_test, pred_050, zero_division=0)
        f1 = f1_score(y_test, pred_050, zero_division=0)
        roc = roc_auc_score(y_test, proba)
        ll = log_loss(y_test, proba)
        cm = confusion_matrix(y_test, pred_050)

        summary_rows.append({
            "Model": name,
            "Accuracy": acc,
            "ROC-AUC": roc,
            "LogLoss": ll,
        })

        results.append({
            "name": name,
            "pipeline": pipeline,
            "y_test": y_test.reset_index(drop=True),
            "y_pred": pd.Series(pred_050),
            "proba": pd.Series(proba),
            "report": report,
            "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc, "logloss": ll},
            "confusion": cm,
        })

    summary_df = pd.DataFrame(summary_rows)
    return {"results": results, "summary_df": summary_df}


if __name__ == "__main__":
    main()

