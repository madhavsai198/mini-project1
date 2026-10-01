import os
import csv
import math
import random

def generate_heart_disease_dataset(n_samples=1200, seed=42):
    """
    Generates a realistic clinical heart disease dataset based on UCI standards using standard library.
    """
    random.seed(seed)
    
    headers = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    rows = []
    
    for _ in range(n_samples):
        age = random.randint(29, 77)
        sex = 1 if random.random() < 0.68 else 0
        cp = random.choices([0, 1, 2, 3], weights=[0.47, 0.17, 0.28, 0.08])[0]
        
        trestbps = int(min(max(random.gauss(131.6, 17.5), 94), 200))
        chol = int(min(max(random.gauss(246.3, 51.8), 126), 564))
        fbs = 1 if random.random() < 0.15 else 0
        restecg = random.choices([0, 1, 2], weights=[0.49, 0.48, 0.03])[0]
        
        base_thalach = 220 - age
        thalach = int(min(max(base_thalach * random.uniform(0.65, 0.95), 71), 202))
        
        exang = 1 if random.random() < 0.32 else 0
        oldpeak = round(min(random.expovariate(1.0), 6.2), 1)
        slope = random.choices([0, 1, 2], weights=[0.45, 0.46, 0.09])[0]
        ca = random.choices([0, 1, 2, 3], weights=[0.58, 0.21, 0.13, 0.08])[0]
        thal = random.choices([1, 2, 3], weights=[0.06, 0.55, 0.39])[0]
        
        # Clinical risk logit formula
        logit = (
            0.04 * (age - 50) +
            0.5 * sex +
            -0.8 * cp +
            0.015 * (trestbps - 120) +
            0.005 * (chol - 200) +
            -0.03 * (thalach - 150) +
            0.9 * exang +
            0.7 * oldpeak +
            0.6 * ca +
            (0.8 if thal == 3 else 0.0) - 1.2
        )
        
        prob = 1.0 / (1.0 + math.exp(-logit))
        target = 1 if prob > 0.5 else 0
        
        rows.append([
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal, target
        ])
        
    return headers, rows

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'heart_disease_data.csv')
    
    headers, rows = generate_heart_disease_dataset(1200)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"[+] Dataset successfully created at: {csv_path}")
    print(f"[+] Samples: {len(rows)}, Features: {len(headers) - 1}")

