import os
import sys
import json
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Preprocessing imports
from preprocessing import run_feature_preparation_pipeline, NUMERICAL_FEATURES


class LogisticRegressionClassifier:
    def __init__(self, lr=0.08, epochs=1200):
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

        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=6, min_samples_split=4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _entropy(self, y):
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split:
            leaf_value = 1 if np.mean(y) >= 0.5 else 0
            leaf_proba = float(np.mean(y)) if len(y) > 0 else 0.0
            return DecisionNode(value=leaf_value, proba=leaf_proba)

        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            leaf_value = 1 if np.mean(y) >= 0.5 else 0
            leaf_proba = float(np.mean(y)) if len(y) > 0 else 0.0
            return DecisionNode(value=leaf_value, proba=leaf_proba)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return DecisionNode(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1.0
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root).value for x in X])

    def predict_proba(self, X):
        return np.array([self._traverse_tree(x, self.root).proba for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestClassifier:
    def __init__(self, n_trees=20, max_depth=6, min_samples_split=4):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = len(X)
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def calculate_evaluation_metrics(y_true, y_pred, y_proba):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

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
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    }


def run_training_pipeline():
    print("[*] Executing Preprocessing Pipeline...")
    X_train, X_test, y_train, y_test, scaler = run_feature_preparation_pipeline()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("[STEP 1] MODEL INITIALIZATION & TRAINING")
    print("=" * 70)
    print(f"[*] Training Dataset Size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"[*] Testing Dataset Size:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"[*] Reproducible Random State: 42")

    models = {
        "logistic_regression": LogisticRegressionClassifier(lr=0.08, epochs=1200),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_split=4),
        "random_forest": RandomForestClassifier(n_trees=20, max_depth=6, min_samples_split=4)
    }

    evaluations = {}

    for name, model in models.items():
        print(f"\n[+] Fitting Model: {name.upper()}")
        
        # Fit ONLY on X_train, y_train
        model.fit(X_train, y_train)
        
        # Predict ONLY on X_test
        y_proba_test = model.predict_proba(X_test)
        y_pred_test = model.predict(X_test)

        # Evaluate on X_test
        metrics = calculate_evaluation_metrics(y_test, y_pred_test, y_proba_test)
        evaluations[name] = metrics

        # Save individual model pickle artifact
        model_filename = f"{name}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"    - Saved artifact to: models/{model_filename}")
        print(f"    - Test Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        print(f"    - Test F1-Score:  {metrics['f1_score']:.4f}")
        print(f"    - Test ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Save evaluation summary
    eval_path = os.path.join(models_dir, 'evaluation_results.json')
    best_name = max(evaluations, key=lambda k: evaluations[k]['f1_score'])
    
    with open(eval_path, 'w') as f:
        json.dump({
            "best_model": best_name,
            "models": evaluations
        }, f, indent=4)

    # Save best model copy
    best_model_obj = models[best_name]
    with open(os.path.join(models_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model_obj, f)

    print("\n" + "=" * 70)
    print("[SUMMARY] TRAINING SUMMARY & MODEL ARTIFACT CHECK")
    print("=" * 70)
    print(f"  - Saved models/logistic_regression.pkl: YES")
    print(f"  - Saved models/decision_tree.pkl:       YES")
    print(f"  - Saved models/random_forest.pkl:       YES")
    print(f"  - Saved models/best_model.pkl:          YES ({best_name.upper()})")
    print("=" * 70)

    return evaluations


if __name__ == '__main__':
    run_training_pipeline()
