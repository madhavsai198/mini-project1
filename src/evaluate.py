import numpy as np


def calculate_evaluation_metrics(y_true, y_pred, y_proba):
    """
    Computes standard classification evaluation metrics:
    Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Trapezoidal ROC-AUC calculation
    sorted_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
    else:
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
        "confusion_matrix": {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        }
    }
