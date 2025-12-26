import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# import helper from models_report
from models_report import collect_results


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(12, 10))

    # Draw heatmap
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=1,
        linecolor="white",
        cbar=True,
        square=True,
        annot_kws={"size": 16, "weight": "bold"}
    )

    # Labels formatting
    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)
    n = cm.shape[0]
    # Ensure provided class names match the confusion matrix size
    if isinstance(class_names, (list, tuple)) and len(class_names) >= n:
        names = class_names[:n]
    else:
        # Fallback to numeric labels
        names = [str(i) for i in range(n)]

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(names, fontsize=14, rotation=45, ha="right")
    ax.set_yticklabels(names, fontsize=14, rotation=0)

    # Title removed (matches reference image)
    plt.tight_layout()

    # Save high-quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_models_comparison(df_metrics: pd.DataFrame, out_path: str, dpi: int = 300):
    # Use a clean, colorblind-friendly seaborn style
    plt.style.use("seaborn-v0_8-colorblind")
    sns.set_style("whitegrid")

    models = df_metrics["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)

    bars_acc = ax.bar(x - width, df_metrics["Accuracy"], width, label="Accuracy", alpha=0.9)
    bars_spec = ax.bar(x, df_metrics["Specificity"], width, label="Specificity", alpha=0.9)
    bars_sens = ax.bar(x + width, df_metrics["Sensitivity"], width, label="Sensitivity", alpha=0.9)

    # Labels and fonts
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 1.05)

    # Grid and spines
    ax.grid(axis='y', linestyle='--', linewidth=0.6, color='lightgray')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # place legend on the right
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=14)

    # annotate bars with percent labels (e.g., 85%)
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


if __name__ == "__main__":
    # Collect results (this will train models if not already trained)
    data = collect_results()
    results = data.get("results", [])
    summary_df = data.get("summary_df", pd.DataFrame())

    if not results:
        print("No model results available to visualize.")
        raise SystemExit(1)

    # Choose model for confusion matrix: pick the model with highest Accuracy from summary
    if not summary_df.empty:
        best_idx = summary_df["Accuracy"].idxmax()
        best_model_name = summary_df.loc[best_idx, "Model"]
    else:
        # fallback to first result
        best_model_name = results[0]["name"]

    # find corresponding result
    best_result = None
    for r in results:
        if r["name"] == best_model_name:
            best_result = r
            break
    if best_result is None:
        best_result = results[0]

    # Build confusion matrix (ensure using predicted labels at 0.5)
    y_true = best_result["y_test"].values
    y_pred = best_result["y_pred"].values

    # Use provided class names mapping (order must match label encoding)
    class_names = ["Normal", "Ulcer", "Low-risk", "High-risk", "Cancer"]

    out_cm = os.path.join(os.getcwd(), "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=out_cm)
    print("Figure 1 saved -> confusion_matrix.png")
    print("Figure 1 - Confusion matrix of the evaluated multiclass model.")

    # Build per-model metrics: Accuracy, Specificity, Sensitivity
    rows = []
    for r in results:
        name = r["name"]
        y_t = r["y_test"].values
        y_p = r["y_pred"].values
        # Prefer stored confusion matrix if available
        cm_r = r.get("confusion")
        if cm_r is None:
            cm_r = confusion_matrix(y_t, y_p)
        # Try binary confusion matrix unpack
        if cm_r.size == 4:
            tn, fp, fn, tp = cm_r.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            # multiclass: compute average specificity per class (macro)
            # Specificity for class i = sum of true negatives for class i / (sum of true negatives + false positives for class i)
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
        rows.append({"Model": name, "Accuracy": float(accuracy), "Specificity": float(specificity), "Sensitivity": float(sensitivity)})

    df_metrics = pd.DataFrame(rows)

    out_bar = os.path.join(os.getcwd(), "models_compare.png")
    plot_models_comparison(df_metrics, out_bar)
    print("Figure 2 saved -> models_compare.png")
    print("Figure 2 - Performance comparison of classification models.")
