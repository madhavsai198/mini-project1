import os
import sys
import json
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless script execution
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import run_feature_preparation_pipeline, NUMERICAL_FEATURES
from train import LogisticRegressionClassifier, DecisionTreeClassifier, DecisionNode, RandomForestClassifier


def calculate_metrics_dict(y_true, y_pred, y_proba):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Calculate ROC-AUC curve points and trapezoidal area
    sorted_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    tpr_list = [0.0]
    fpr_list = [0.0]
    accum_tp = 0
    accum_fp = 0
    for label in y_true_sorted:
        if label == 1:
            accum_tp += 1
        else:
            accum_fp += 1
        tpr_list.append(accum_tp / n_pos)
        fpr_list.append(accum_fp / n_neg)
        
    roc_auc = 0.0
    for i in range(1, len(fpr_list)):
        roc_auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0
    roc_auc = float(roc_auc)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr_list": fpr_list,
        "tpr_list": tpr_list
    }


calculate_evaluation_metrics = calculate_metrics_dict


def generate_evaluation_charts(results_dict, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    # 1. ROC Curve Comparison Plot
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {'logistic_regression': '#2563eb', 'decision_tree': '#e11d48', 'random_forest': '#059669'}
    
    for m_name, m_data in results_dict.items():
        auc_val = m_data['roc_auc']
        c = colors.get(m_name, '#4b5563')
        ax.plot(m_data['fpr_list'], m_data['tpr_list'], label=f"{m_name.replace('_', ' ').title()} (AUC = {auc_val:.4f})", color=c, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance Baseline (AUC = 0.5000)')
    ax.set_title('ROC Curve Model Comparison (Unseen Test Set)', fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(figures_dir, 'roc_curve_comparison.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 2. Confusion Matrices Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for idx, (m_name, m_data) in enumerate(results_dict.items()):
        cm = m_data['confusion_matrix']
        matrix_arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        ax = axes[idx]
        cax = ax.matshow(matrix_arr, cmap='Blues')
        ax.set_title(m_name.replace('_', ' ').title(), fontsize=10, fontweight='bold', pad=15)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Not Placed (0)', 'Placed (1)'])
        ax.set_yticklabels(['Not Placed (0)', 'Placed (1)'])
        
        for i in range(2):
            for j in range(2):
                val = matrix_arr[i, j]
                color = "white" if val > np.max(matrix_arr)/2 else "black"
                ax.text(j, i, str(val), ha='center', va='center', color=color, fontweight='bold')

    plt.tight_layout()
    cm_path = os.path.join(figures_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()

    return roc_path, cm_path


def run_evaluation_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')

    # Load unseen test data
    _, X_test, _, y_test, scaler = run_feature_preparation_pipeline()

    model_names = ['logistic_regression', 'decision_tree', 'random_forest']
    results = {}

    print("=" * 70)
    print("[EVALUATION STEP 1] EVALUATING TRAINED MODELS ON UNSEEN TEST DATA (199 Students)")
    print("=" * 70)

    for m_name in model_names:
        model_path = os.path.join(models_dir, f"{m_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact missing: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        metrics = calculate_metrics_dict(y_test, y_pred, y_proba)
        results[m_name] = metrics

    # Print Comparison Table
    print("\n[EVALUATION STEP 2] MODEL COMPARISON TABLE")
    print("-" * 75)
    print(f"{'Model Algorithm':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)

    best_model_name = None
    best_f1 = -1.0

    for m_name, m_data in results.items():
        print(f"{m_name.replace('_', ' ').title():<22} | {m_data['accuracy']*100:>8.2f}% | {m_data['precision']:>10.4f} | {m_data['recall']:>10.4f} | {m_data['f1_score']:>10.4f} | {m_data['roc_auc']:>10.4f}")
        
        if m_data['f1_score'] > best_f1:
            best_f1 = m_data['f1_score']
            best_model_name = m_name

    print("-" * 75)
    print(f"[BEST MODEL SELECTED]: {best_model_name.replace('_', ' ').title()} (Highest Test F1: {best_f1:.4f})")

    # Generate Evaluation Charts
    roc_p, cm_p = generate_evaluation_charts(results, figures_dir)
    print(f"\n[+] Generated ROC Curve Comparison Plot: {roc_p}")
    print(f"[+] Generated Confusion Matrices Plot:   {cm_p}")

    # Save Model Config
    config_data = {
        "best_model_name": best_model_name,
        "best_model_path": f"models/{best_model_name}.pkl",
        "eval_metrics": results[best_model_name],
        "test_sample_count": len(y_test)
    }
    
    config_path = os.path.join(models_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"[+] Model Configuration file saved to: {config_path}")
    print("=" * 70)

    return results, best_model_name


if __name__ == '__main__':
    run_evaluation_pipeline()
