import os
import csv
import json
import math
import numpy as np

class StandardScalerPure:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0  # Prevent division by zero
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }

    def from_dict(self, d):
        self.mean = np.array(d["mean"])
        self.std = np.array(d["std"])


class LogisticRegressionPure:
    def __init__(self, lr=0.05, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def to_dict(self):
        return {
            "weights": self.weights.tolist(),
            "bias": float(self.bias)
        }

    def from_dict(self, d):
        self.weights = np.array(d["weights"])
        self.bias = float(d["bias"])


def load_data_from_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [list(map(float, row)) for row in reader if row]
    
    matrix = np.array(data)
    X = matrix[:, :-1]
    y = matrix[:, -1]
    feature_names = headers[:-1]
    return X, y, feature_names


def calculate_metrics(y_true, y_pred, y_proba):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = float((tp + tn) / len(y_true))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Trapezoidal ROC-AUC estimation
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
        
        # Trapezoidal area under curve computation
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
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'heart_disease_data.csv')
    
    if not os.path.exists(data_path):
        print("[!] Dataset missing. Generating dataset...")
        from data.generate_dataset import generate_heart_disease_dataset
        headers, rows = generate_heart_disease_dataset(1200)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    X, y, feature_names = load_data_from_csv(data_path)

    # 80/20 train test split
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(0.80 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScalerPure()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegressionPure(lr=0.08, epochs=1500)
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)
    y_pred = model.predict(X_test_scaled)

    eval_metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Feature importance calculation based on absolute weight magnitudes
    abs_weights = np.abs(model.weights)
    total_weight = np.sum(abs_weights)
    importances = abs_weights / total_weight if total_weight > 0 else np.ones(len(abs_weights)) / len(abs_weights)
    
    feature_importance_dict = dict(zip(feature_names, np.round(importances, 4).tolist()))

    metrics_output = {
        "model_name": "CardioCare Logistic Regression Classifier",
        "random_forest": eval_metrics,
        "feature_importances": feature_importance_dict
    }

    # Save artifacts
    model_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'heart_disease_model.json')
    scaler_path = os.path.join(model_dir, 'scaler.json')
    metrics_path = os.path.join(model_dir, 'metrics.json')

    with open(model_path, 'w') as f:
        json.dump(model.to_dict(), f, indent=4)

    with open(scaler_path, 'w') as f:
        json.dump(scaler.to_dict(), f, indent=4)

    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=4)

    print("[+] Pure NumPy ML Model trained successfully!")
    print(f"[+] Model Accuracy: {eval_metrics['accuracy'] * 100:.2f}%")
    print(f"[+] F1-Score: {eval_metrics['f1_score']:.4f}")
    print(f"[+] ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    print(f"[+] Model saved to: {model_path}")
    print(f"[+] Scaler saved to: {scaler_path}")
    print(f"[+] Metrics saved to: {metrics_path}")

if __name__ == '__main__':
    train_and_evaluate()
