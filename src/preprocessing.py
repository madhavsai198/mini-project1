import os
import csv
import numpy as np
import pickle

FEATURE_NAMES = [
    'ssc_percentage', 'hsc_percentage', 'cgpa', 'coding_rating',
    'aptitude_score', 'soft_skills_score', 'internships',
    'projects_completed', 'workshops_attended', 'backlogs', 'gender_encoded'
]

TARGET_NAME = 'placement_status'


class StandardStudentScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0  # Prevent division by zero
        return (X - self.mean) / self.std

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transforming!")
        return (X - self.mean) / self.std


def load_raw_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset file not found: {filepath}")

    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader if row]

    # Feature mapping index dictionary
    col_idx = {name: i for i, name in enumerate(headers)}
    
    rows_processed = []
    labels = []

    for row in data:
        gender_val = 1 if row[col_idx['gender']] == 'Male' else 0
        ssc = float(row[col_idx['ssc_percentage']])
        hsc = float(row[col_idx['hsc_percentage']])
        cgpa = float(row[col_idx['cgpa']])
        coding = float(row[col_idx['coding_rating']])
        apt = float(row[col_idx['aptitude_score']])
        soft = float(row[col_idx['soft_skills_score']])
        intern = float(row[col_idx['internships']])
        proj = float(row[col_idx['projects_completed']])
        work = float(row[col_idx['workshops_attended']])
        backlogs = float(row[col_idx['backlogs']])
        
        target = 1 if row[col_idx['placement_status']] == 'Placed' else 0

        feat_vector = [ssc, hsc, cgpa, coding, apt, soft, intern, proj, work, backlogs, gender_val]
        rows_processed.append(feat_vector)
        labels.append(target)

    return np.array(rows_processed), np.array(labels), FEATURE_NAMES


def run_preprocessing_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'student_data.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_raw, y_raw, feature_names = load_raw_csv(raw_path)
    print(f"[*] Raw dataset loaded successfully. Shape: {X_raw.shape}")

    scaler = StandardStudentScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Save cleaned csv file
    processed_csv = os.path.join(processed_dir, 'cleaned_data.csv')
    scaler_pkl = os.path.join(models_dir, 'scaler.pkl')

    with open(processed_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + [TARGET_NAME])
        for i in range(len(X_scaled)):
            row = list(X_scaled[i]) + [int(y_raw[i])]
            writer.writerow(row)

    with open(scaler_pkl, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"[+] Cleaned scaled data saved to: {processed_csv}")
    print(f"[+] Scaler artifact saved to: {scaler_pkl}")
    return X_scaled, y_raw, scaler


if __name__ == '__main__':
    run_preprocessing_pipeline()
