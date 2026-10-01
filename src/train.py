import os
import sys
import csv
import json
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import run_preprocessing_pipeline, FEATURE_NAMES
from evaluate import calculate_evaluation_metrics


class LogisticRegressionClassifier:
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
    def __init__(self, max_depth=5, min_samples_split=4):
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
    def __init__(self, n_trees=15, max_depth=5, min_samples_split=4):
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


def train_and_compare_models():
    print("[*] Preprocessing raw dataset...")
    X, y, scaler = run_preprocessing_pipeline()

    # Train / Test split (80/20)
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(0.80 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    models = {
        "logistic_regression": LogisticRegressionClassifier(lr=0.08, epochs=1200),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_split=4),
        "random_forest": RandomForestClassifier(n_trees=20, max_depth=6, min_samples_split=4)
    }

    evaluations = {}
    best_score = -1.0
    best_model_name = None
    best_model_obj = None

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')

    print("\n[*] Training & Evaluating Models on Test Set:")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        metrics = calculate_evaluation_metrics(y_test, y_pred, y_proba)
        evaluations[name] = metrics

        print(f"Model: {name.upper()}")
        print(f"  - Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall:    {metrics['recall']:.4f}")
        print("-" * 60)

        # Save individual model pickle artifact
        model_pkl_path = os.path.join(models_dir, f"{name}.pkl")
        with open(model_pkl_path, 'wb') as f:
            pickle.dump(model, f)

        # Objective criteria for best model selection: highest F1-score
        if metrics['f1_score'] > best_score:
            best_score = metrics['f1_score']
            best_model_name = name
            best_model_obj = model

    print(f"[BEST MODEL] BEST MODEL SELECTED DYNAMICALLY: {best_model_name.upper()} (F1: {best_score:.4f})")

    # Save best model artifact
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_obj, f)

    # Save summary evaluation json
    eval_json_path = os.path.join(models_dir, 'evaluation_results.json')
    with open(eval_json_path, 'w') as f:
        json.dump({
            "best_model": best_model_name,
            "models": evaluations
        }, f, indent=4)

    print(f"[+] Evaluation results saved to: {eval_json_path}")
    return evaluations, best_model_name


if __name__ == '__main__':
    train_and_compare_models()
