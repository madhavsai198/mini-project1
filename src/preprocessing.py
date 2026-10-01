import os
import csv
import pickle
import numpy as np

# Feature Definitions
NUMERICAL_FEATURES = [
    'ssc_percentage', 'hsc_percentage', 'cgpa', 'coding_rating',
    'aptitude_score', 'soft_skills_score', 'internships',
    'projects_completed', 'workshops_attended', 'backlogs'
]

CATEGORICAL_FEATURES = ['gender']
TARGET_COLUMN = 'placement_status'


class ModularColumnTransformerScaler:
    """
    StandardScaler and Feature Transformer.
    Calculates mean (mu) and std (sigma) strictly on training data to prevent data leakage.
    """
    def __init__(self):
        self.num_means = None
        self.num_stds = None
        self.feature_names_out = None

    def fit_transform(self, X_dict_list):
        """
        Fits mean and std on numerical features of training fold and transforms features.
        """
        X_matrix, self.feature_names_out = self._extract_matrix(X_dict_list)
        
        # Separate numerical and categorical columns
        num_indices = [i for i, name in enumerate(self.feature_names_out) if name in NUMERICAL_FEATURES]
        
        num_data = X_matrix[:, num_indices]
        self.num_means = np.mean(num_data, axis=0)
        self.num_stds = np.std(num_data, axis=0)
        self.num_stds[self.num_stds == 0] = 1.0  # Prevent division by zero

        X_transformed = np.copy(X_matrix)
        X_transformed[:, num_indices] = (num_data - self.num_means) / self.num_stds
        return X_transformed

    def transform(self, X_dict_list):
        """
        Transforms test fold using mean and std learned strictly from training fold.
        """
        if self.num_means is None or self.num_stds is None:
            raise ValueError("Transformer must be fitted on X_train before transforming X_test!")

        X_matrix, _ = self._extract_matrix(X_dict_list)
        num_indices = [i for i, name in enumerate(self.feature_names_out) if name in NUMERICAL_FEATURES]
        
        X_transformed = np.copy(X_matrix)
        num_data = X_matrix[:, num_indices]
        X_transformed[:, num_indices] = (num_data - self.num_means) / self.num_stds
        return X_transformed

    def _extract_matrix(self, dict_list):
        feature_names = NUMERICAL_FEATURES + ['gender_encoded']
        matrix = []
        for d in dict_list:
            gender_val = 1 if d.get('gender', 'Male') == 'Male' else 0
            row = [float(d[f]) for f in NUMERICAL_FEATURES] + [gender_val]
            matrix.append(row)
        return np.array(matrix), feature_names


def load_dataset_for_ml(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file missing: {csv_path}")

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [r for r in reader if r]

    col_idx = {name: i for i, name in enumerate(headers)}
    
    X_dict_list = []
    y_labels = []

    for row in data:
        # Separate Student_ID (dropped from model features)
        record = {h: row[col_idx[h]] for h in headers if h != 'student_id' and h != TARGET_COLUMN}
        target_val = int(row[col_idx[TARGET_COLUMN]]) if str(row[col_idx[TARGET_COLUMN]]).isdigit() else (1 if row[col_idx[TARGET_COLUMN]] == 'Placed' else 0)
        
        X_dict_list.append(record)
        y_labels.append(target_val)

    return X_dict_list, np.array(y_labels)


def stratified_train_test_split(X_dict_list, y_array, test_size=0.20, random_state=42):
    """
    Performs stratified train/test split to preserve target class proportions (Placed vs Not Placed).
    """
    np.random.seed(random_state)
    
    idx_class_0 = np.where(y_array == 0)[0]
    idx_class_1 = np.where(y_array == 1)[0]
    
    np.random.shuffle(idx_class_0)
    np.random.shuffle(idx_class_1)
    
    test_cnt_0 = int(len(idx_class_0) * test_size)
    test_cnt_1 = int(len(idx_class_1) * test_size)
    
    test_indices = np.concatenate([idx_class_0[:test_cnt_0], idx_class_1[:test_cnt_1]])
    train_indices = np.concatenate([idx_class_0[test_cnt_0:], idx_class_1[test_cnt_1:]])
    
    np.random.shuffle(test_indices)
    np.random.shuffle(train_indices)
    
    X_train_dict = [X_dict_list[i] for i in train_indices]
    X_test_dict = [X_dict_list[i] for i in test_indices]
    
    y_train = y_array[train_indices]
    y_test = y_array[test_indices]
    
    return X_train_dict, X_test_dict, y_train, y_test


def run_feature_preparation_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("[STEP 1] LOADING DATASET & SEPARATING X (FEATURES) vs y (TARGET)")
    print("=" * 70)
    X_dict_list, y = load_dataset_for_ml(csv_path)
    
    print(f"[*] Total Samples: {len(X_dict_list)}")
    print(f"[*] Feature Count (X): {len(X_dict_list[0])} (Student_ID dropped)")
    print(f"[*] Target Variable (y): {len(y)} samples (Placed: {np.sum(y == 1)}, Not Placed: {np.sum(y == 0)})")

    # Stratified Train-Test Split
    print("\n[STEP 2] PERFORMING STRATIFIED TRAIN / TEST SPLIT (80% Train, 20% Test)")
    print("-" * 70)
    X_train_dict, X_test_dict, y_train, y_test = stratified_train_test_split(
        X_dict_list, y, test_size=0.20, random_state=42
    )

    print(f"[*] Train Set Size: {len(X_train_dict)} samples | Target Split: Placed={np.sum(y_train==1)}, Not Placed={np.sum(y_train==0)} ({np.mean(y_train)*100:.1f}% Placed)")
    print(f"[*] Test Set Size:  {len(X_test_dict)} samples | Target Split: Placed={np.sum(y_test==1)}, Not Placed={np.sum(y_test==0)} ({np.mean(y_test)*100:.1f}% Placed)")

    # Fit Preprocessing Pipeline ONLY on Training Data
    print("\n[STEP 3] FITTING PREPROCESSING TRANSFORMER STRICTLY ON X_train")
    print("-" * 70)
    scaler = ModularColumnTransformerScaler()
    X_train_scaled = scaler.fit_transform(X_train_dict)
    
    # Transform Test Data using parameters learned from Training Data
    X_test_scaled = scaler.transform(X_test_dict)
    
    print("[+] Feature scaling complete:")
    print(f"    - X_train_scaled shape: {X_train_scaled.shape}")
    print(f"    - X_test_scaled shape:  {X_test_scaled.shape}")
    print("    - Data Leakage Status: STRICTLY PREVENTED (Mean & Std computed on X_train only)")

    # Save fitted scaler artifact
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"[+] Scaler artifact saved to: {scaler_path}")
    print("=" * 70)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == '__main__':
    run_feature_preparation_pipeline()
